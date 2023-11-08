import os

from setuptools import setup

WITHIN_DOCKER_ENV = os.getenv("WITHIN_DOCKER_ENV", False)
print(WITHIN_DOCKER_ENV)

if WITHIN_DOCKER_ENV:
    use_scm_version = None
else:
    use_scm_version = {"local_scheme": "no-local-version"}

with open("requirements.txt") as f:
    required = f.read().splitlines()

if __name__ == "__main__":
    setup(
        use_scm_version=use_scm_version,
        install_requires=required,
        test_suite="test",
    )
