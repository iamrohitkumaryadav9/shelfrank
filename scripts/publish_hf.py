"""Write the Hugging Face model card, and upload the fine-tuned encoder only when asked.

    python scripts/publish_hf.py                          write docs/hf_model_card.md
    python scripts/publish_hf.py --repo USER/NAME --yes   create a public model repository and upload

Needs `huggingface-cli login` (a write token) for the upload. Nothing is uploaded without --yes.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from shelfrank.config import REPO_ROOT
from shelfrank.dense.hf_card import card

MODEL = REPO_ROOT / "artifacts/models/bge_small_esci"
FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "modules.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
    "training.json",
)
REPOSITORY = "iamrohitkumaryadav9/shelfrank"


def load(name: str) -> dict:
    """A results file."""
    return json.loads((REPO_ROOT / f"results/{name}.json").read_text(encoding="utf-8"))


def main() -> None:
    """Card always; upload on request."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    text = card(load("finetune_esci"), load("dense_dev"), load("final_run"), REPOSITORY)
    (REPO_ROOT / "docs/hf_model_card.md").write_text(text, encoding="utf-8", newline="\n")
    print("wrote docs/hf_model_card.md")
    if not args.repo:
        return
    if not args.yes:
        raise SystemExit("add --yes to publish")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        for name in FILES:
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(MODEL / name, out / name)
        (out / "README.md").write_text(text, encoding="utf-8")
        api.upload_folder(
            folder_path=str(out),
            repo_id=args.repo,
            commit_message="Fine-tuned encoder and model card",
        )
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
