from deep.gectorPredict.predict import (
    predict_for_paragraph,
    replacements_to_json,
)
from deep.gectorPredict.gector.gec_model import GecBERTModel
from django.http import JsonResponse
import os
from kernel.settings import BASE_DIR

### Initializing the model

MIN_ERR_PROB = {"all": 0.7, "comma": 0.8}
ADD_CONF = 0.3
TOKEN_METH = "split+spacy"
if os.environ.get("min_err_prob_all") and os.environ.get("min_err_prob_comma"):
    MIN_ERR_PROB["all"] = float(os.environ.get("min_err_prob_all"))
    MIN_ERR_PROB["comma"] = float(os.environ.get("min_err_prob_comma"))
if os.environ.get("add_conf"):
    ADD_CONF = float(os.environ.get("add_conf"))
if os.environ.get("tokenizer_method"):
    TOKEN_METH = str(os.environ.get("tokenizer_method"))

args = {
    "vocab_path": os.path.join(BASE_DIR, "deep/gectorPredict/MODEL_DIR/vocabulary"),
    "model_path": [os.path.join(BASE_DIR, "deep/gectorPredict/MODEL_DIR/best.th")],
    "max_len": 50,
    "min_len": 2,
    "iteration_count": 5,
    "min_error_probability": MIN_ERR_PROB,
    "lowercase_tokens": 0,
    "transformer_model": "bertimbaubase",
    "special_tokens_fix": 1,
    "additional_confidence": ADD_CONF,
    "is_ensemble": 0,
    "weights": None,
}

model = GecBERTModel(
    model_paths=args["model_path"],
    vocab_path=args["vocab_path"],
    max_len=args["max_len"],
    min_len=args["min_len"],
    iterations=args["iteration_count"],
    min_error_probability=args["min_error_probability"],
    lowercase_tokens=args["lowercase_tokens"],
    model_name=args["transformer_model"],
    special_tokens_fix=args["special_tokens_fix"],
    log=False,
    confidence=args["additional_confidence"],
    is_ensemble=args["is_ensemble"],
    weigths=args["weights"],
)


def output(request):
    # obtaining the query parameter 'text'
    if request.method == "GET":
        request_string = str(request.GET.get("text"))
    if request.method == "POST":
        request_string = str(request.POST.get("text"))

    # making inference
    repl = predict_for_paragraph(
        request_string,
        model,
        tokenizer_method=TOKEN_METH,
    )
    json_output = replacements_to_json(
        version="1.0",
        request_string=request_string,
        replacements_dictionary=repl,
    )

    return JsonResponse(json_output, json_dumps_params={"ensure_ascii": False})
