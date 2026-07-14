# Music-Activity

## Privacy Notes

- The app is currently client-side only. Generated audio and prompt text stay in the browser unless you add a backend later.
- The page still makes a few third-party requests:
  - `cdn.jsdelivr.net` for the Transformers.js module import
  - Hugging Face model endpoints for MusicGen weights
  - Google Fonts for the current font and icon assets
- Vercel deployment is configured with baseline security headers in `vercel.json`.
- If you want a stricter privacy posture, the next step would be to self-host fonts/icons and remove any remaining external assets.

## Model Attribution

- Audio generation uses `Xenova/musicgen-small`, a browser-compatible ONNX release of Meta AI's `facebook/musicgen-small`.
- Both model pages are licensed under `CC BY-NC 4.0`: [Xenova/musicgen-small](https://huggingface.co/Xenova/musicgen-small) and [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small).
- Acoustix uses the model as provided and does not modify the weights.
