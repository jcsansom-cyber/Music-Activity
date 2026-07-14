# Music-Activity

## Privacy Notes

- The app is currently client-side only. Generated audio and prompt text stay in the browser unless you add a backend later.
- The page still makes a few third-party requests:
  - `cdn.jsdelivr.net` for the Transformers.js module import
  - Hugging Face model endpoints for MusicGen weights
  - Google Fonts for the current font and icon assets
- Vercel deployment is configured with baseline security headers in `vercel.json`.
- If you want a stricter privacy posture, the next step would be to self-host fonts/icons and remove any remaining external assets.
