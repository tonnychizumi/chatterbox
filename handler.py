import runpod
import torch
import torchaudio
import base64
import tempfile
import os

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(
    device=device,
    t3_model="v3"
)


def handler(job):
    job_input = job.get("input", {})

    text = job_input.get("text")
    language_id = job_input.get("language_id", "sw")
    exaggeration = float(job_input.get("exaggeration", 0.7))
    cfg_weight = float(job_input.get("cfg_weight", 0.3))
    reference_audio_base64 = job_input.get("reference_audio")

    if not text:
        return {"error": "text is required"}

    reference_path = None
    output_path = None

    try:
        if reference_audio_base64:
            reference_bytes = base64.b64decode(reference_audio_base64)

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as ref_file:
                ref_file.write(reference_bytes)
                reference_path = ref_file.name

        generate_kwargs = {
            "language_id": language_id,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        }

        if reference_path:
            generate_kwargs["audio_prompt_path"] = reference_path

        wav = model.generate(
            text[:300],
            **generate_kwargs
        )

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as output_file:
            output_path = output_file.name

        torchaudio.save(
            output_path,
            wav.cpu(),
            model.sr
        )

        with open(output_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(
                audio_file.read()
            ).decode("utf-8")

        return {
            "audio": audio_base64,
            "format": "wav",
            "sample_rate": model.sr,
            "language_id": language_id,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if reference_path and os.path.exists(reference_path):
            os.remove(reference_path)

        if output_path and os.path.exists(output_path):
            os.remove(output_path)


runpod.serverless.start({"handler": handler})
