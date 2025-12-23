#!/usr/bin/env python3

import getpass
import argparse
from argparse import Namespace, ArgumentParser
import os
import sys
import re
import requests
from typing import Dict, Tuple, Optional, Match
from dataclasses import dataclass
from openai import OpenAI
import google.generativeai as genai
from google.generativeai import types
from anthropic import Anthropic, APIStatusError

NORMALIZATION_INSTRUCTION:str = "Rewrite the following sentences to a clear, fixed point form, such that if I asked you to rewrite again nothing would change. All aspects must be included in the rewritten form. Add background knowledge if necessary to clarify the scenario. Provide all the rewritten sentences as plain text, no explanations or summary."
LOGIC_FORMAT_INSTRUCTION:str = "Use annotated tff formulae. Every sentence must be translated. Questions must be translated into formulae with the 'conjecture' role. All other sentences must be translated into formulae with the 'axiom' role. All the necessary type declarations must be provided. Declare each type in a separate annotated formula. Remember that variables start uppercase, and all other symbols start lowercase. Remember that the connectives are ~ for negation, | for disjunction, & for conjunction, => for implication, <=> for equivalence, = for equality, != for inequality. Put parentheses around all binary formulae. Output only the TPTP TFF result, no explanations and no comments lines. Use plain text, not markdown."
SUMO_TERM_REQUEST:str = "Use symbols from the SUMO ontology."
NL2L_INSTRUCTION:str = "Translate this English into TPTP typed first-order logic." + LOGIC_FORMAT_INSTRUCTION 
L2NL_INSTRUCTION:str = "Translate this TPTP typed first-order logic into natural English. Formulae with the 'conjecture' role must be expressed as questions. All other formulae must be expressed as statements of fact. Provide only the English as plain text, one sentence per line, no blank lines, no explanations or summary."
SIMLARITY_INSTRUCTION:str = "Tell me how similar in meaning these two test segments are, as a real number in the range 0.0 to 1.0. Output only the number on the first line, then a explanation of any differences on following lines."

DEFAULT_OPENAI_MODEL:str = "gpt-5-chat-latest"
OPENAI_API_KEY:str = ""

DEFAULT_GOOGLE_MODEL:str = "gemini-2.5-flash"
GOOGLE_API_KEY:str = ""

DEFAULT_ANTHROPIC_MODEL:str = "claude-sonnet-4-5"
ANTHROPIC_API_KEY:str = ""

DEFAULT_NORMALIZATION_MODEL = "OpenAI---" + DEFAULT_OPENAI_MODEL
DEFAULT_NL2L_MODEL = "OpenAI---" + DEFAULT_OPENAI_MODEL
DEFAULT_L2NL_MODEL = "OpenAI---" + DEFAULT_OPENAI_MODEL
DEFAULT_SIMILARITY_MODEL = "OpenAI---" + DEFAULT_OPENAI_MODEL

SYSTEM_ON_TPTP_URL:str = "https://tptp.org/cgi-bin/SystemOnTPTPFormReply"
SYNTAX_CHECKER:str = "TPTP4X---"
SYNTAX_ERROR_RE:str = r"^(ERROR:.*)"
SYNTAX_ERROR_INSTRUCTION:str = "There is a syntax error in that TPTP typed first-order logic. Here is the error message and the logic that has the errors. Please try again to correct the error. " + LOGIC_FORMAT_INSTRUCTION

TYPE_CHECKER:str = "Leo-III-STC---"
TYPE_ERROR_RE:str = r"SZS status [a-zA-Z]*Error.*? : (.*)"
# TYPE_ERROR_RE:str = r"SZS status [a-zA-Z]*Error.*? : (.*)|.*(Interpreter error in annotated TPTP.*)"
TYPE_ERROR_INSTRUCTION:str = "There is a type error in that TPTP typed first-order logic. Here is the error message and the logic that has the errors. Please try again to correct the error. " + LOGIC_FORMAT_INSTRUCTION

SIMILARITY_ERROR_INSTRUCTION:str = "That logic has a very different meaning to the original text. Please try again to make the logic have the same meaning as the sentences." + LOGIC_FORMAT_INSTRUCTION

SZSAbbreviations = {
    "Theorem": "THM",
    "CounterSatisfiable": "CSA",
    "Unsatisfiable": "UNS",
    "Satisfiable": "SAT",
    "ContradictoryAxioms": "CAX",
    "Timeout": "TMO",
    "GaveUp": "GUP",
    "Unknown": "UNK"
}

QUIETNESS:int = 0
#--------------------------------------------------------------------------------------------------
@dataclass
class ZigZagResultType:
    Converged:bool = False
    ZigZagNumber:int = 0
    LastTwoSimilarityScore:float = 0.0
    PenultimateText:str = ""
    LastTwoDifferences:str = ""
    FinalText:str = ""
    OriginalSimilarityScore:float = 0.0
    OriginalDifferences:str = ""
    Logic:str = ""
    SyntaxCorrections:int = 0
    TypeCorrections:int = 0
    SimilarityCorrections:int = 0
    SZSStatus:str = "NON"
    SZSOutput:str = ""
#--------------------------------------------------------------------------------------------------
def QuietPrint(Level:int,Indent:int,Message:str,End="\n") -> None:

    if abs(Level) >= QUIETNESS:
        print(" " * Indent,end='')
        if Level == 0:
            print("%---- DEBUG -----")
        elif Level == 1:
            print("%---- DETAIL ----")
        if Level >= 2:
            print("% ",end='')
        print(Message,end=End)
        if Level == 0 or Level == 1:
            print("%------------------")
#--------------------------------------------------------------------------------------------------
def SetModel(ModelRequested:str) -> str:

    if re.search(r".*---.*",ModelRequested):
        return ModelRequested
    else:
        if ModelRequested == "Google":
            return "Google---" + DEFAULT_GOOGLE_MODEL
        elif ModelRequested == "Anthropic":
            return "Anthropic---" + DEFAULT_ANTHROPIC_MODEL
        else:
            return "OpenAI---" + DEFAULT_OPENAI_MODEL
#--------------------------------------------------------------------------------------------------
def GetModels(CommandLineArguments:Namespace,APIKeyLines:str) -> tuple[str,str,str,str]:

#----API keys are global
    global OPENAI_API_KEY
    global GOOGLE_API_KEY
    global ANTHROPIC_API_KEY

    NormalizationModel:str = ""
    NL2LModel:str = ""
    L2NLModel:str = ""
    SimilarityModel:str = ""
    FileModel:str = ""
#---Can't declare Matches:re.Match = None

#----First look if a key is specified in the file
    Matches = re.search(r"^#\s*([A-Z]+)_API_KEY\s*=\s*(.+)",APIKeyLines,re.MULTILINE)
    if Matches:
        if Matches.group(1) == "OPENAI":
            FileModel = "OpenAI"
            OPENAI_API_KEY = Matches.group(2)
        elif Matches.group(1) == "GOOGLE":
            FileModel = "Google"
            GOOGLE_API_KEY = Matches.group(2)
        elif Matches.group(1) == "ANTHROPIC":
            FileModel = "Anthropic"
            ANTHROPIC_API_KEY = Matches.group(2)
        else:
            print(f"ERROR: {Matches.group(1)} is not a known model")
            sys.exit(0)
        NormalizationModel = SetModel(FileModel)
        NL2LModel = SetModel(FileModel)
        L2NLModel = SetModel(FileModel)
        SimilarityModel = SetModel(FileModel)
#----Otherwise use non-file model
    else:
#----Set for all models, either from argument or defaults
        if hasattr(CommandLineArguments,"model"):
            NormalizationModel = SetModel(CommandLineArguments.model)
            NL2LModel = SetModel(CommandLineArguments.model)
            L2NLModel = SetModel(CommandLineArguments.model)
            SimilarityModel = SetModel(CommandLineArguments.model)
        else:
            NormalizationModel = SetModel(DEFAULT_NORMALIZATION_MODEL)
            NL2LModel = SetModel(DEFAULT_NL2L_MODEL)
            L2NLModel = SetModel(DEFAULT_L2NL_MODEL)
            SimilarityModel = SetModel(DEFAULT_SIMILARITY_MODEL)
#----Override for individual uses
        if hasattr(CommandLineArguments,"normalization_model"):
            NormalizationModel = SetModel(CommandLineArguments.normalization_model)
        if hasattr(CommandLineArguments,"nl2l_model"):
            NL2LModel = SetModel(CommandLineArguments.nl2l_model)
        if hasattr(CommandLineArguments,"l2nl_model"):
            L2NLModel = SetModel(CommandLineArguments.l2nl_model)
        if hasattr(CommandLineArguments,"similarity_model"):
            SimilarityModel = SetModel(CommandLineArguments.similarity_model)

#----Check user has the necessary API keys
    if (re.match(r"^OpenAI",NormalizationModel) or re.match(r"^OpenAI",NL2LModel) or \
re.match(r"^OpenAI",L2NLModel) or re.match(r"^OpenAI",SimilarityModel)) and \
len(OPENAI_API_KEY) == 0:
        if "OPENAI_API_KEY" in os.environ:
            OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY"))
        else:
            print("ERROR: OpenAI API key not in the file or environment variable OPENAI_API_KEY")
            sys.exit(0)

    if (re.match(r"^Google",NormalizationModel) or re.match(r"^Google",NL2LModel) or \
re.match(r"^Google",L2NLModel) or re.match(r"^Google",SimilarityModel)) and \
len(GOOGLE_API_KEY) == 0:
        if "GOOGLE_API_KEY" in os.environ:
            GOOGLE_API_KEY = str(os.getenv("GOOGLE_API_KEY"))
        else:
            print("ERROR: Google API key not in the file or environment variable GOOGLE_API_KEY")
            sys.exit(0)

#----Make sure there is no mess in the output
    if (re.match(r"^Google",NormalizationModel) or re.match(r"^Google",NL2LModel) or \
re.match(r"^Google",L2NLModel) or re.match(r"^Google",SimilarityModel)) and \
(not "GRPC_VERBOSITY" in os.environ or os.getenv("GRPC_VERBOSITY") != "NONE"):
        print("To suppress GRPC messages set your GRPC_VERBOSITY environment variable to NONE")

    if (re.match(r"^Anthropic",NormalizationModel) or re.match(r"^Anthropic",NL2LModel) or \
re.match(r"^Anthropic",L2NLModel) or re.match(r"^Anthropic",SimilarityModel)) and \
len(ANTHROPIC_API_KEY) == 0:
        if "ANTHROPIC_API_KEY" in os.environ:
            ANTHROPIC_API_KEY = str(os.getenv("ANTHROPIC_API_KEY"))
        else:
            print(
"ERROR: Anthropic API key not in the file or environment variable ANTHROPIC_API_KEY")
            sys.exit(0)

    return NormalizationModel,NL2LModel,L2NLModel,SimilarityModel
#--------------------------------------------------------------------------------------------------
# Requires environment variable ANTHROPIC_API_KEY
def CallAnthropic(Instruction:str,Content:str,ModelName:str) -> str:

    Prompt:str = Instruction + "\n" + Content

    Client = Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        Message = Client.messages.create(
            model=ModelName,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": Prompt,
                }
            ]
        )
        if hasattr(Message.content[0],"text"):
            return Message.content[0].text
        else:
            return ""
    except Exception as e:
        print(f"ERROR: Calling Anthropic: {e}")
        sys.exit(0)
#--------------------------------------------------------------------------------------------------
# Requires environment variable OPENAI_API_KEY
def CallOpenAI(Instruction:str,Content:str,ModelName:str) -> str:

    Client = OpenAI(api_key=OPENAI_API_KEY)

# print(f"Calling OpenAI with\n{Instruction}\n{Content}")
    try:
        OpenAIResponse = Client.chat.completions.create(
            model = ModelName,
            temperature=0,
            messages=[
                {"role": "system", "content": Instruction},
                {"role": "user", "content": Content},
            ],
        )
        return OpenAIResponse.choices[0].message.content or ""
    except Exception as e:
        print(f"ERROR: Calling OpenAI: {e}")
        sys.exit(0)
#--------------------------------------------------------------------------------------------------
# Requires environment variable GOOGLE_API_KEY
def CallGoogle(Instruction:str,Content:str,ModelName:str) -> str:

    GoogleResponse:Optional[types.GenerateContentResponse] = None
    Prompt:str = Instruction + "\n" + Content

#----Set your API key
    genai.configure(api_key=GOOGLE_API_KEY)
#----Call the model
    Model = genai.GenerativeModel(ModelName)
    try:
        GoogleResponse = Model.generate_content(Prompt)
        return GoogleResponse.text
    except Exception as e:
        print(f"ERROR: Calling Google: {e}")
        sys.exit(0)
#--------------------------------------------------------------------------------------------------
def CallLLM(Model:str,Content:str,Task:str) -> str:

    Result:str = ""
    Instruction:str = ""
    Matches:Optional[Match[str]] = None

    if Task == "NORMALIZATION":
        Instruction = NORMALIZATION_INSTRUCTION
    elif Task == "NL2L":
        Instruction = NL2L_INSTRUCTION
    elif Task == "L2NL":
        Instruction = L2NL_INSTRUCTION
    elif Task == "SyntaxError":
        Instruction = SYNTAX_ERROR_INSTRUCTION
    elif Task == "TypeError":
        Instruction = TYPE_ERROR_INSTRUCTION
    elif Task == "SimilarityError":
        Instruction = SIMILARITY_ERROR_INSTRUCTION
    elif Task == "Similarity":
        Instruction = SIMLARITY_INSTRUCTION
    else:
        QuietPrint(0,0,"Invalid LLM task " + Task)
        return "None"

    Matches = re.search("(.*)---(.*)",Model)
    if Matches:
        if Matches.group(1) == "OpenAI":
            Result = CallOpenAI(Instruction,Content,Matches.group(2))
        elif Matches.group(1) == "Google":
            Result = CallGoogle(Instruction,Content,Matches.group(2))
        elif Matches.group(1) == "Anthropic":
            Result = CallAnthropic(Instruction,Content,Matches.group(2))
        else:
            QuietPrint(0,0,"Unknown model " + Model)
            return "None"
    else:
        QuietPrint(0,0,"Unknown model " + Model)
        return "None"

    return Result
#--------------------------------------------------------------------------------------------------
def CompareNLs(Model:str,Text1:str,Text2:str) -> tuple[float,str]:

    Content:str = ""
    SimilarityResponse:Optional[str] = ""
    Similarity:float = 0.0
    SimilarityMatch:Optional[re.Match] = None

    Content = "Here is the first text segment:\n" + Text1 + "\n" + \
              "Here is the second text segment:\n" + Text2 + "\n"
    SimilarityResponse = CallLLM(Model,Content,"Similarity")
    if SimilarityResponse:
        QuietPrint(0,0,f"The {Model} similarity output is {SimilarityResponse}")
        FirstLine,Separator,Differences = SimilarityResponse.partition("\n")
        Differences = Differences.strip()
        SimilarityMatch = re.search(r"[-+]?\d*\.?\d+",FirstLine)
        if SimilarityMatch:
            Similarity = float(SimilarityMatch.group(0))
            QuietPrint(1,0,f"The similarity is {Similarity:.2f}")
            QuietPrint(1,0,f"The differences are " + Differences)
            return Similarity,Differences
    return 0.0,"No similarity computed"
#--------------------------------------------------------------------------------------------------
def SystemOnTPTP(Logic:str,System:str,TimeLimit:int,ErrorRE:str) -> tuple[bool,str,str]:

    DEFAULT_URL_PARAMETERS:Dict[str,Tuple[Optional[str],str]] = {
        "NoHTML": (None, '1'),
        "QuietFlag": (None, '-q1'),
        "SubmitButton": (None,'RunSelectedSystems'),
        "ProblemSource": (None,'UPLOAD')
    }

    URLParameters:Dict[str,Tuple[Optional[str],str]] = DEFAULT_URL_PARAMETERS
#---Can't declare Matches:re.Match = None
    SystemOnTPTPResponse = None

    URLParameters["TimeLimit___"+System] = (None,str(TimeLimit))
    URLParameters["System___"+System] = (System,System)
    URLParameters["UPLOADProblem"] = (Logic,Logic)
    SystemOnTPTPResponse = requests.post(SYSTEM_ON_TPTP_URL,files=URLParameters)

    QuietPrint(0,0,f"The output from {System} is\n{SystemOnTPTPResponse.text}")
#----If looking for errors
    if ErrorRE != "None":
        Matches = re.search(ErrorRE,SystemOnTPTPResponse.text,re.MULTILINE)
        if Matches:
            QuietPrint(1,0,f"Syntax error {Matches.group(1)}")
            return True,"Error",Matches.group(1)
        else:
            QuietPrint(1,0,"Syntax passed:\n" + SystemOnTPTPResponse.text)
            Matches = re.search("START OF SYSTEM OUTPUT\n(.+?)END OF SYSTEM OUTPUT", \
SystemOnTPTPResponse.text,re.DOTALL)
            if Matches:
                return False,"Success",Matches.group(1).strip()
            else:
                return False,"Success","Missing output from " + System
#----Else running ATP system, use SZS standards
    else:
        Matches = re.search(r"SZS status\s+(\S+)",SystemOnTPTPResponse.text,re.MULTILINE)
        if Matches:
            SZSStatus = Matches.group(1)
            QuietPrint(1,0,"% SZS status: " + SZSStatus)
            StartPattern = r".*SZS output start.*"
            EndPattern = r".*SZS output end.*"
            Matches = re.search("SZS output start[^\n]*\n(.+?)\n[^\n]*SZS output end", \
SystemOnTPTPResponse.text,re.DOTALL)
            if Matches:
                return True,SZSStatus,Matches.group(1).strip()
            else:
                return True,SZSStatus,"Missing solution from " + System
        else:
            return False,"NoSuccess","Nothing done by " + System
        
#--------------------------------------------------------------------------------------------------
def RunATP(Logic:str,Prover:str,ModelFinder:str,TimeLimit:int) -> tuple[str,str]:

    GotSZS:bool = False
    System:str = ""
    SZSStatus:str = ""
    SZSOutput:str = ""
#---Can't declare Matches:re.Match = None

    Matches = re.search(", *conjecture *,",Logic,re.MULTILINE)
    if Matches:
        if Prover != "None":
            QuietPrint(4,0,f"Running {Prover} to try prove the conjecture")
            System = Prover
        elif ModelFinder != "None":
            QuietPrint(4,0,f"Not running {ModelFinder} as there is a conjecture")
            System = "None"
        else:
            System = "None"
    else:
        if ModelFinder != "None":
            QuietPrint(4,0,f"Running {ModelFinder} to test the consistency of the axioms")
            System = ModelFinder
        elif Prover != "None":
            QuietPrint(4,0,f"Running {Prover} to test the inconsistency of the axioms")
            System = Prover
        else:
            System = "None"

    if System != "None":
        GotSZS,SZSStatus,SZSOutput = SystemOnTPTP(Logic,System,TimeLimit,"None")
        if GotSZS:
            SZSStatus = SZSAbbreviations[SZSStatus]
            QuietPrint(2,2,f"{System} SZS status {SZSStatus}")
            QuietPrint(2,2,f"{System} SZS output start")
            QuietPrint(-2,2,SZSOutput)
            QuietPrint(2,2,f"{System} SZS output end")
            return SZSStatus,SZSOutput

    return "NON","None"
#--------------------------------------------------------------------------------------------------
def ParseCommandLine() -> Namespace:

    Parser:argparse.ArgumentParser = argparse.ArgumentParser("Natural language to Logic converter")

    Parser.add_argument("-q","--quietness",type=int,default=3,
        help="Output suppression, 0=none,5=max. Default %(default)s")
    Parser.add_argument("-M","--model",type=str,default=argparse.SUPPRESS,
        help="Model for normalization, NL2L, L2NL, and similarity. \
Format is Company[---Model], Company is OpenAI or Google or Anthropic, ---Model is optional.")
    Parser.add_argument("-F","--normalization_model",type=str,default=argparse.SUPPRESS,
        help="Model for NL normalization. Default " + DEFAULT_NORMALIZATION_MODEL)
    Parser.add_argument("-N","--nl2l_model",type=str,default=argparse.SUPPRESS,
        help="Model for NL to L conversion. Default " + DEFAULT_NL2L_MODEL)
    Parser.add_argument("-L","--l2nl_model",type=str,default=argparse.SUPPRESS,
        help="Model for L to NL conversion. Default " + DEFAULT_L2NL_MODEL)
    Parser.add_argument("-S","--similarity_model",type=str,default=argparse.SUPPRESS,
        help="Model for NL similarity measurement. Default " + DEFAULT_SIMILARITY_MODEL)
    Parser.add_argument("-n","--no_normalize",action="store_false",default=False,
        help="Don't normalize the text. Default %(default)s.")
    Parser.add_argument("-a","--similarity_acceptance",type=float,default=0.74,
        help="Similarity required to accept one ZigZag. Default %(default)s.")
    Parser.add_argument("-c","--similarity_convergence",type=float,default=0.94,
        help="Similarity required to stop ZigZaging. Default %(default)s.")
    Parser.add_argument("-s","--zigzagging_acceptable",type=float,default=0.94,
        help="Similarity required to stop repeating ZigZaging. Default %(default)s.")
    Parser.add_argument("-z","--zigzag_limit",type=int,default=10,
        help="Maximal ZigZags. Default %(default)s.")
    Parser.add_argument("-l","--zig_correction_limit",type=int,default=10,
        help="Maximal NL2L corrections in one Zig(Zag). Default %(default)s.")
    Parser.add_argument("-r","--zigzagging_limit",type=int,default=3,
        help="Maximal ZigZaging repeats. Default %(default)s.")
    Parser.add_argument("-p","--prover",type=str,default="None",
        help="ATP system for THM/UNS. Default %(default)s.")
    Parser.add_argument("-m","--model_finder",type=str,default="None",
        help="ATP system for CSA/SAT. Default %(default)s.")
    Parser.add_argument("-t","--time_limit",type=int,default=10,
        help="ATP system time limit. Default %(default)s.")
    Parser.add_argument("-v","--values",action="store_true",default=False,
        help="Print summary as comma separated values. Default %(default)s.")
    Parser.add_argument("NL_file_name",
        help="The file containing the natural language")

    return Parser.parse_args()
#--------------------------------------------------------------------------------------------------
def ZigZagToConvergence(CommandLineArguments:Namespace,NormalizationModel:str,NL2LModel:str,
L2NLModel:str,SimilarityModel:str,OriginalText:str,NormalizeText:bool,ATPTimeLimit:int) -> \
ZigZagResultType:

#----These are set by command line parameters
    SimilarityAcceptable:float = 0.0
    SimilarityConverged:float = 0.0
    ZigZagLimit:int = 0
    ZigCorrectionLimit:int = 0

    SimilarityAcceptable = CommandLineArguments.similarity_acceptance
    SimilarityConverged = CommandLineArguments.similarity_convergence
    ZigZagLimit = CommandLineArguments.zigzag_limit
    ZigCorrectionLimit = CommandLineArguments.zig_correction_limit

#----This is where the real action starts
    OldText:str = ""
    NewText:str = ""
    Logic:str = ""
    Converged:bool = False
    LastTwoSimilarityScore:float = 0.0
    OriginalSimilarityScore:float = 0.0
    LastTwoDifferences:str = ""
    OriginalDifferences:str = ""
    TPTPSyntaxError:bool = False
    TPTPTypeError:bool = True
    SimilarityError:bool = True
    ZigZagNumber:int = 0
    SyntaxCheckNumber:int = 0;
    TypeCheckNumber:int = 0;
    SimilarityCheckNumber:int = 0
    ZigCorrectionNumber:int = 0
    TotalSyntaxCorrections:int = 0
    TotalTypeCorrections:int = 0
    TotalSimilarityCorrections:int = 0
    SZSStatus:str = ""
    SZSOutput:str = ""

    NewText = OriginalText
#----Loop until two most recent NLs have enough similarity
    while not Converged and ZigZagNumber < ZigZagLimit and \
ZigCorrectionNumber < ZigCorrectionLimit:
        OldText = NewText
        ZigZagNumber += 1
        ZigCorrectionNumber = 0
        QuietPrint(4,0,f"Doing Zig (NL2L) number {ZigZagNumber}")
        Logic = CallLLM(NL2LModel,OldText,"NL2L")
#----Loop while latest NL has meaning too different from the original
        SimilarityCheckNumber = 0
        SimilarityError = True
        while SimilarityError and ZigCorrectionNumber < ZigCorrectionLimit:
            QuietPrint(1,2,f"The logic from {NL2LModel} at NL2L number {ZigZagNumber} is\n" + Logic)
#----Loop while there is a type error (or something like that)
            TypeCheckNumber = 0
            TPTPTypeError = True
            while TPTPTypeError and ZigCorrectionNumber < ZigCorrectionLimit:
#----Loop while there is a syntax error in the logic
                SyntaxCheckNumber = 1
                QuietPrint(4,2,f"Doing syntax check {SyntaxCheckNumber} after NL2L {ZigZagNumber}")
                TPTPSyntaxError,SZSStatus,SZSOutput = SystemOnTPTP(Logic,SYNTAX_CHECKER,\
ATPTimeLimit,SYNTAX_ERROR_RE)
                while TPTPSyntaxError and ZigCorrectionNumber < ZigCorrectionLimit:
                    QuietPrint(2,4,f"Syntax error {SyntaxCheckNumber} in NL2L {ZigZagNumber}")
                    QuietPrint(2,4,f"{SZSStatus}:{SZSOutput}")
                    ZigCorrectionNumber += 1
                    TotalSyntaxCorrections += 1
                    QuietPrint(4,4,f"Doing Zig (NL2L) for syntax error {SyntaxCheckNumber} \
after NL2L {ZigZagNumber}")
                    Logic = CallLLM(NL2LModel,SZSOutput + "\n" + Logic,"SyntaxError")
                    QuietPrint(1,4,f"The syntax corrected logic from {NL2LModel} is\n{Logic}")
                    SyntaxCheckNumber += 1
                    QuietPrint(4,4,f"Doing syntax check {SyntaxCheckNumber} after NL2L \
{ZigZagNumber}")
                    TPTPSyntaxError,SZSStatus,SZSOutput = SystemOnTPTP(Logic,SYNTAX_CHECKER,\
ATPTimeLimit,SYNTAX_ERROR_RE)
                if not TPTPSyntaxError:
                    Logic = SZSOutput
                    TypeCheckNumber += 1
                    QuietPrint(4,2,f"Doing type check {TypeCheckNumber} after NL2L {ZigZagNumber}")
                    TPTPTypeError,SZSStatus,SZSOutput = SystemOnTPTP(Logic,TYPE_CHECKER,\
ATPTimeLimit,TYPE_ERROR_RE)
                    if TPTPTypeError and ZigCorrectionNumber < ZigCorrectionLimit:
                        QuietPrint(2,4,f"Type error {TypeCheckNumber} in NL2L {ZigZagNumber}")
                        QuietPrint(2,4,f"{SZSStatus}:{SZSOutput}")
                        ZigCorrectionNumber += 1
                        TotalTypeCorrections += 1
                        QuietPrint(4,4,f"Doing Zig (NL2L) for type error {TypeCheckNumber} \
after NL2L {ZigZagNumber}")
                        Logic = CallLLM(NL2LModel,SZSOutput + "\n" + Logic,"TypeError")
                        QuietPrint(1,4,f"The type corrected logic from {NL2LModel} is\n{Logic}")
            if not TPTPTypeError:
                QuietPrint(4,2,f"Made logic without syntax or type errors at NL2L number \
{ZigZagNumber}")
                QuietPrint(4,0,f"Doing Zag (L2NL) number {ZigZagNumber}")
                NewText = CallLLM(L2NLModel,Logic,"L2NL")
                QuietPrint(1,2,f"The language from {L2NLModel} at L2NL number {ZigZagNumber} is\n \
{NewText}")
#                if NormalizeText:
#                    NewText = CallLLM(NormalizationModel,NewText,"NORMALIZATION")
#                    QuietPrint(1,2,f"The normalized language from {L2NLModel} at L2NL number \
#{ZigZagNumber} is\n {NewText}")
                SimilarityCheckNumber += 1
                QuietPrint(4,2,f"Doing original similarity check {SimilarityCheckNumber} \
after L2NL {ZigZagNumber}")
                OriginalSimilarityScore,OriginalDifferences = \
CompareNLs(SimilarityModel,OriginalText,NewText)
                SimilarityError = OriginalSimilarityScore < SimilarityAcceptable
#----There is a similarity error, rerun the NL2L
                if SimilarityError and ZigCorrectionNumber < ZigCorrectionLimit:
                    QuietPrint(4,4,f"NL 0~{ZigZagNumber} == {OriginalSimilarityScore:.2f} < \
{SimilarityAcceptable:.2f}. Zig again.")
                    ZigCorrectionNumber += 1
                    TotalSimilarityCorrections += 1
                    QuietPrint(4,2,f"Doing Zig (NL2L) for similarity error {SimilarityCheckNumber} \
after NL2L {ZigZagNumber}")
                    Logic = CallLLM(NL2LModel,OldText,"SimilarityError")
                    QuietPrint(1,4,f"The similarity corrected logic from {NL2LModel} is\n{Logic}")
                else:
                    QuietPrint(4,4,f"NL {ZigZagNumber-1}~{ZigZagNumber} == \
{OriginalSimilarityScore:.2f} >= {SimilarityAcceptable:.2f}. ZigZag ok.")
#----Now we have acceptable new language, or we bailed due to too many corrections
        if ZigCorrectionNumber <= ZigCorrectionLimit:
            if not SimilarityError:
                QuietPrint(4,2,f"Doing convergence similarity check {ZigZagNumber}")
                LastTwoSimilarityScore,LastTwoDifferences = \
CompareNLs(SimilarityModel,OldText,NewText)
                Converged = LastTwoSimilarityScore > SimilarityConverged
                if Converged:
                    QuietPrint(4,2,f"NL {ZigZagNumber-1}~{ZigZagNumber} == \
{LastTwoSimilarityScore:.2f} >= {SimilarityConverged:.2f}. ZigZag converged.")
                else:
                    QuietPrint(4,4,f"NL {ZigZagNumber-1}~{ZigZagNumber} == \
{LastTwoSimilarityScore:.2f} < {SimilarityConverged:.2f}. ZigZag again.")
                QuietPrint(2,0,f"There were {ZigCorrectionNumber} corrections in ZigZag \
{ZigZagNumber}")
        else: 
            QuietPrint(4,2,f"{ZigCorrectionNumber} corrections exceeds limit {ZigCorrectionLimit}.")

    return ZigZagResultType(Converged,ZigZagNumber,LastTwoSimilarityScore,OldText,\
LastTwoDifferences,NewText,OriginalSimilarityScore,OriginalDifferences,Logic,\
TotalSyntaxCorrections,TotalTypeCorrections,TotalSimilarityCorrections)

#--------------------------------------------------------------------------------------------------
def PrintResult(FilePath:str,OriginalText:str,NormalizedText:str,ZigZagRepeats:int,
ZigZagResult:ZigZagResultType,NormalizationModel:str,NL2LModel:str,L2NLModel:str,
SimilarityModel:str):

    QuietPrint(5,0,"-------------------------------------------------------------------------")
    QuietPrint(5,0,f"Converged at ZigZag number {ZigZagRepeats}:{ZigZagResult.ZigZagNumber} \
with similarity {ZigZagResult.LastTwoSimilarityScore:.2f}")
    QuietPrint(5,0,f"The original and final NL have similarity: \
{ZigZagResult.OriginalSimilarityScore:.2f}")
    QuietPrint(5,0,f"Normalization model: {NormalizationModel}")
    QuietPrint(5,0,f"NL2L model:          {NL2LModel}")
    QuietPrint(5,0,f"L2NL model:          {L2NLModel}")
    QuietPrint(5,0,f"Similarity model:    {SimilarityModel}")
    QuietPrint(5,0,"-------------------------------------------------------------------------")
    QuietPrint(5,0,f"The original NL is :\n{OriginalText}")
    QuietPrint(5,0,"-------------------------------------------------------------------------")
    QuietPrint(5,0,f"The normalized NL is :\n{NormalizedText}")
    QuietPrint(5,0,"-------------------------------------------------------------------------")
    QuietPrint(5,0,f"The final NL is :\n{ZigZagResult.FinalText}")
    QuietPrint(5,0,"-------------------------------------------------------------------------")
    QuietPrint(5,0,f"The differences between the original and final NL are:\n\
{ZigZagResult.OriginalDifferences}")
    QuietPrint(5,0,"-------------------------------------------------------------------------")
    QuietPrint(2,0,f"The penultimate NL is :\n{ZigZagResult.PenultimateText}")
    QuietPrint(2,0,"-------------------------------------------------------------------------")
    QuietPrint(2,0,f"The differences between the penultimate and final NL are:\n \
{ZigZagResult.LastTwoDifferences}")
    QuietPrint(2,0,"-------------------------------------------------------------------------")
    QuietPrint(5,0,f"The final logic is :")
    QuietPrint(-6,0,f"{ZigZagResult.Logic}")
    QuietPrint(5,0,"-------------------------------------------------------------------------")
    QuietPrint(5,0,f"The ATP SZS status is {ZigZagResult.SZSStatus} with output\n\
{ZigZagResult.SZSOutput}")
    QuietPrint(5,0,"-------------------------------------------------------------------------")
    QuietPrint(4,0,f"{ZigZagResult.SyntaxCorrections}/{ZigZagResult.TypeCorrections}/\
{ZigZagResult.SimilarityCorrections} syntax/type/similarity corrections")
    QuietPrint(4,0,"-------------------------------------------------------------------------")

#--------------------------------------------------------------------------------------------------
def main():

#----Quietness is global
    global QUIETNESS

    CommandLineArguments:Namespace = Namespace()
    FilePath:str = ""
    Prover:str = ""
    ModelFinder:str = ""
    ATPTimeLimit:int = 0
    ZigZaggingAcceptable:float = 0.0
    ZigZaggingLimit:int = 0
    PrintCSV:bool = False
    NormalizeText:bool = True

    NormalizationModel:str = ""
    NL2LModel:str = ""
    L2NLModel:str = ""
    SimilarityModel:str = ""
    OriginalText:str = ""
    APIKeyLines:str = ""
    ZigZagRepeats:int = 0
    BestZigZagSimilarity:float = 0.0

#----Results from one ZigZag sequence
    ZigZagResult:ZigZagResultType = ZigZagResultType()
    BestZigZagResult:ZigZagResultType = ZigZagResultType()

    CommandLineArguments = ParseCommandLine()
    QUIETNESS = CommandLineArguments.quietness
    FilePath = CommandLineArguments.NL_file_name
    Prover = CommandLineArguments.prover
    ModelFinder = CommandLineArguments.model_finder
    ATPTimeLimit = CommandLineArguments.time_limit
    ZigZaggingAcceptable = CommandLineArguments.zigzagging_acceptable
    ZigZaggingLimit = CommandLineArguments.zigzagging_limit
    PrintCSV = CommandLineArguments.values
    NormalizeText = not CommandLineArguments.no_normalize

    with open(FilePath,"r",encoding="utf-8") as FileHandle:
        for FileLine in FileHandle:
            FileLine.strip()
            if FileLine.startswith('#'):
                if re.match(r".*API_KEY\s*=",FileLine):
                    APIKeyLines += FileLine
            else:
               OriginalText += FileLine
    OriginalText = OriginalText.strip()

    NormalizationModel,NL2LModel,L2NLModel,SimilarityModel = GetModels(\
CommandLineArguments,APIKeyLines)

    if NormalizeText:
        NormalizedText = CallLLM(NormalizationModel,OriginalText,"NORMALIZATION")
        QuietPrint(3,0,"-------------------------------------------------------------------------")
        QuietPrint(3,0,f"Original text:\n{OriginalText}")
        QuietPrint(3,0,"-------------------------------------------------------------------------")
        QuietPrint(3,0,f"NormalizedText text:\n{NormalizedText}")
        QuietPrint(3,0,"-------------------------------------------------------------------------")
    else:
        NormalizedText = OriginalText

    ZigZagRepeats = 0
    BestZigZagSimilarity = 0.0
    while ZigZagRepeats < ZigZaggingLimit and \
BestZigZagResult.OriginalSimilarityScore < ZigZaggingAcceptable:
        ZigZagRepeats += 1
        QuietPrint(4,0,f"---- ZigZag sequence {ZigZagRepeats} to improve on original similarity \
{BestZigZagResult.OriginalSimilarityScore:.2f}, need similarity {ZigZaggingAcceptable:.2f}")
        ZigZagResult = ZigZagToConvergence(CommandLineArguments,NormalizationModel,NL2LModel,
L2NLModel,SimilarityModel,NormalizedText,NormalizeText,ATPTimeLimit)
        if ZigZagResult.Converged and \
ZigZagResult.OriginalSimilarityScore > BestZigZagResult.OriginalSimilarityScore:
            ZigZagResult.SZSStatus,ZigZagResult.SZSOutput = \
RunATP(ZigZagResult.Logic,Prover,ModelFinder,ATPTimeLimit)
            BestZigZagResult = ZigZagResult

    if BestZigZagResult.Converged:
        QuietPrint(4,0,"-------------------------------------------------------------------------")
        if BestZigZagResult.OriginalSimilarityScore >= ZigZaggingAcceptable:
            QuietPrint(4,0,f"Acceptable","")
        else:
            QuietPrint(4,0,f"No acceptable","")
        QuietPrint(-4,0,f" convergence after {ZigZagRepeats}:{ZigZaggingLimit} ZigZag sequences")
        QuietPrint(4,0,f"Printing the best ZigZag sequence result")
        PrintResult(FilePath,OriginalText,NormalizedText,ZigZagRepeats,BestZigZagResult,
NormalizationModel,NL2LModel,L2NLModel,SimilarityModel)
    else:
        QuietPrint(4,0,"-------------------------------------------------------------------------")
        QuietPrint(4,0,f"No convergence after {ZigZagRepeats}:{ZigZaggingLimit} ZigZag sequences")
        SZSStatus = "NON"
        QuietPrint(4,0,"-------------------------------------------------------------------------")

    if PrintCSV:
        print(f"RESULT: {FilePath},\
{BestZigZagResult.OriginalSimilarityScore >= ZigZaggingAcceptable},\
{BestZigZagResult.OriginalSimilarityScore:.2f},{ZigZagRepeats},{BestZigZagResult.Converged},\
{BestZigZagResult.LastTwoSimilarityScore:.2f},{BestZigZagResult.ZigZagNumber},\
{BestZigZagResult.SZSStatus},Unknown,{BestZigZagResult.SyntaxCorrections},\
{BestZigZagResult.TypeCorrections},{BestZigZagResult.SimilarityCorrections},{NL2LModel},\
{L2NLModel},{SimilarityModel}")
#--------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#--------------------------------------------------------------------------------------------------
