# Privacy

- The application is local-first and has no authentication, external API, or telemetry service.
- Prediction inputs are sent to the local FastAPI process. The saved history contains only module, class label, optional positive-class score, and timestamp; it does not contain the submitted measurements or personal identifiers.
- Set `HISTORY_ENABLED=false` in the environment, or disable history in Settings, to prevent new history records. Existing records remain until cleared in Prediction History.
- The SQLite file is a convenience feature, not encrypted storage, a secure medical record, or a multi-user store. Do not enter real patient data.
- PDF reports are generated in memory and contain the current submitted feature values. The app does not persist those reports; downloaded files are managed by the user's browser/device.
- Dataset CSVs, joblib models, generated metrics, SQLite history, `.env`, local Python environment, and portable Node runtime are ignored by Git.
- CORS defaults to the local Vite origin. Do not expose the development API to a network or claim production security/compliance.