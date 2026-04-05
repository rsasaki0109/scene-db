"""VLM-based scene captioning using OpenAI API."""

import base64
import json
import os
from pathlib import Path

from scene_db.features import generate_caption as generate_rule_caption


def _get_openai_client():
    """Get OpenAI client if available."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError:
        return None


def _encode_image(image_path: Path) -> str:
    """Encode image to base64 for API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _pick_representative_image(image_paths: list[Path]) -> Path | None:
    """Pick the middle frame as representative."""
    if not image_paths:
        return None
    return image_paths[len(image_paths) // 2]


def generate_vlm_caption(
    image_paths: list[Path],
    avg_speed_kmh: float,
    distance_m: float,
    model: str = "gpt-4o-mini",
) -> str:
    """Generate caption using VLM. Falls back to rule-based if unavailable."""
    client = _get_openai_client()
    if client is None:
        return generate_rule_caption(avg_speed_kmh, distance_m)

    image_path = _pick_representative_image(image_paths)
    if image_path is None or not image_path.exists():
        return generate_rule_caption(avg_speed_kmh, distance_m)

    base64_image = _encode_image(image_path)
    speed_info = f"Vehicle speed: {avg_speed_kmh:.0f} km/h, distance traveled: {distance_m:.1f} m"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this autonomous driving scene in one sentence. "
                                "Focus on: road type, weather, surrounding objects (cars, "
                                "pedestrians, buildings), and driving behavior. "
                                f"Context: {speed_info}"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        caption = response.choices[0].message.content.strip()
        return f"{caption} [{avg_speed_kmh:.0f} km/h, {distance_m:.1f} m]"
    except Exception:
        return generate_rule_caption(avg_speed_kmh, distance_m)
