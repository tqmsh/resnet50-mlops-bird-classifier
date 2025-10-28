import yaml
import os
import re
from typing import Any, Dict

def load_config_with_env_vars(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration with environment variable substitution.

    Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax.
    """
    with open(config_path, 'r') as f:
        content = f.read()

    # Load environment variables from .env file if it exists
    env_file = os.path.join(os.path.dirname(config_path), '..', '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

    # Substitute environment variables
    def replace_env_vars(match):
        var_expression = match.group(1)

        # Check for default value syntax: ${VAR:-default}
        if ':-' in var_expression:
            var_name, default_value = var_expression.split(':-', 1)
        else:
            var_name = var_expression
            default_value = ''

        return os.environ.get(var_name, default_value)

    # Replace ${VAR_NAME} patterns
    pattern = r'\$\{([^}]+)\}'
    content = re.sub(pattern, replace_env_vars, content)

    return yaml.safe_load(content)