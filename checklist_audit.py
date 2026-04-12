with open("inference.py", encoding="utf-8") as f:
    src = f.read()

checks = [
    ("1. API_BASE_URL = os.getenv(... default set)",    'API_BASE_URL = os.getenv("API_BASE_URL", ' in src),
    ("2. MODEL_NAME   = os.getenv(... default set)",    'MODEL_NAME = os.getenv("MODEL_NAME", '   in src),
    ("3. HF_TOKEN     = os.getenv (NO default)",        'HF_TOKEN = os.getenv("HF_TOKEN")'        in src and 'HF_TOKEN = os.getenv("HF_TOKEN", ' not in src),
    ("4. LOCAL_IMAGE_NAME = os.getenv present",         'LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")' in src),
    ("5. from openai import OpenAI used",               "from openai import OpenAI" in src),
    ("6. OpenAI() gets api_key=HF_TOKEN",               "api_key=HF_TOKEN" in src),
    ("7. OpenAI() gets base_url=API_BASE_URL",          "base_url=API_BASE_URL" in src),
    ("8. [START] log emitted",                          "[START]" in src),
    ("9. [STEP]  log emitted",                          "[STEP]"  in src),
    ("10. [END]  log emitted",                          "[END]"   in src),
]

print()
print("=" * 62)
print("  Pre-Submission Checklist -- inference.py audit")
print("=" * 62)
all_ok = True
for label, ok in checks:
    icon = "[PASS]" if ok else "[FAIL]"
    if not ok:
        all_ok = False
    print(f"  {icon}  {label}")

print("=" * 62)
if all_ok:
    print("  RESULT: ALL 10 CHECKS PASSED -- ready to submit!")
else:
    print("  RESULT: SOME CHECKS FAILED")
print("=" * 62)
