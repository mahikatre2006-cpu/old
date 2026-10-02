import os

# Set before pipeline.config loads backend/.env, so default test runs can never touch Neon.
# Real env vars win over .env, and an empty DATABASE_URL means "use SQLite".
os.environ["DATABASE_URL"] = ""
