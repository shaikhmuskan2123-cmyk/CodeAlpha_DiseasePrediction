# Test Report

Environment: Python 3.10.11, project `.venv`, FastAPI TestClient, Node.js v24.20.0 portable runtime, Vite 6.4.4.

## Automated checks

- `python -m pytest backend\tests -q`: **17 passed**, 0 failed, 1 dependency deprecation warning (Starlette/AnyIO).
- `npm run build`: **passed** (TypeScript strict build and Vite production bundle).
- Model training: all three independent pipelines trained and saved from their documented sources. Heart 303 records; diabetes 520; breast cancer 569.

Tests exercise API health and module listing, actual saved pipeline inference for all three modules, output-score bounds, additive Logistic Regression explanation arithmetic, global-only explanation labeling, dataset record counts, invalid/incomplete features, history opt-out, PDF generation and PDF response headers/structure.

## Browser smoke checks

- Dashboard loaded all three real trained models and actual heart metrics.
- Heart (13 features), diabetes (16 features), and breast-cancer (30 nucleus measurements) forms each returned a real saved-pipeline output.
- The heart Logistic Regression screen displayed signed local contributions, intercept, and decision score; category labels reflected UCI definitions.
- Performance page displayed stored test metrics and model comparison chart.
- Dataset explorer showed the 303-record UCI table; search filtering reduced visible rows and enabled filtered CSV export.
- At 390px viewport, the overview remained visible with `document.body.scrollWidth` 375px, so no horizontal overflow was observed.
- API report endpoint returned HTTP 200 and `application/pdf`; the browser automation environment did not surface the frontend blob as a Playwright download event. Temporary smoke predictions were cleared from local history afterward.

## Dependency review

`npm install` after migration to Tailwind 4.1.11 reported **0 vulnerabilities**.

These tests establish software behavior on this machine only, not clinical validity, safety, fairness, production readiness, or regulatory compliance.