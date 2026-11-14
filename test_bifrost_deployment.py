"""
test_bifrost_deployment.py - Unit tests for minimal public Bifrost deployment and app integration.

Verifies:
1. BIFROST_URL normalization (ensuring /v1 endpoint for OpenAI client and clean root for health check).
2. LLM_VIRTUAL_KEY handling from config/secrets (acting as API key and x-bf-vk header).
3. check_gateway_health using BIFROST_URL and virtual key header.
4. Automatic fallback to direct Gemini when Bifrost gateway is unreachable.
5. Deployment artifacts (deploy/bifrost/config.json, Dockerfile, render.yaml) exist and contain zero plaintext secrets.
"""

import json
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import httpx
from openai import APIConnectionError

import llm
from scripts.check_evidence_clean import scan_obj_for_keys

ROOT_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = ROOT_DIR / "deploy" / "bifrost"


class TestBifrostDeploymentAndAppIntegration(unittest.TestCase):
    """Test suite for public Bifrost gateway deployment and resilient app routing."""

    def test_bifrost_url_normalizes_and_resolves_base_url(self):
        """BIFROST_URL should normalize to /v1 for client base_url and root URL for health checks."""
        # Case A: URL without /v1
        with patch("llm.load_dotenv"), patch.dict(
            os.environ,
            {"BIFROST_URL": "https://my-bifrost.onrender.com", "USE_BIFROST": "true"},
            clear=False,
        ):
            self.assertTrue(llm.is_use_bifrost())
            root_url = llm.get_gateway_root_url()
            self.assertEqual(root_url, "https://my-bifrost.onrender.com")

            _, base_url, _ = llm._resolve_task_models_from_env()
            self.assertEqual(base_url, "https://my-bifrost.onrender.com/v1")

        # Case B: URL with trailing slash or already has /v1
        with patch("llm.load_dotenv"), patch.dict(
            os.environ,
            {"BIFROST_URL": "https://my-bifrost.onrender.com/v1/", "USE_BIFROST": "true"},
            clear=False,
        ):
            root_url = llm.get_gateway_root_url()
            self.assertEqual(root_url, "https://my-bifrost.onrender.com")

            _, base_url, _ = llm._resolve_task_models_from_env()
            self.assertEqual(base_url, "https://my-bifrost.onrender.com/v1")

    def test_llm_virtual_key_as_api_key_and_header(self):
        """LLM_VIRTUAL_KEY should serve as api_key when LLM_API_KEY is not set, and inject x-bf-vk header."""
        with patch("llm.load_dotenv"), patch.dict(
            os.environ,
            {
                "USE_BIFROST": "true",
                "LLM_API_KEY": "",
                "LLM_VIRTUAL_KEY": "sk-bf-stale-detector-test-vk",
                "BIFROST_URL": "https://my-bifrost.onrender.com",
            },
            clear=False,
        ):
            api_key, base_url, model_chain = llm.get_llm_config()
            self.assertEqual(api_key, "sk-bf-stale-detector-test-vk")

            client, _, client_base_url = llm.get_llm_client()
            self.assertEqual(client_base_url, "https://my-bifrost.onrender.com/v1")
            self.assertEqual(client.default_headers.get("x-bf-vk"), "sk-bf-stale-detector-test-vk")

    def test_check_gateway_health_with_bifrost_url_and_virtual_key(self):
        """check_gateway_health should probe /health and attach x-bf-vk header."""
        ok_response = httpx.Response(
            200,
            json={"status": "ok", "version": "1.3.0", "providers": ["gemini", "openai"]},
        )
        with patch("llm.load_dotenv"), patch.dict(
            os.environ,
            {
                "BIFROST_URL": "https://my-bifrost.onrender.com",
                "LLM_VIRTUAL_KEY": "sk-bf-test-vk",
                "USE_BIFROST": "true",
            },
            clear=False,
        ):
            with patch("llm.httpx.get", return_value=ok_response) as mock_get:
                health = llm.check_gateway_health()

            self.assertTrue(health["healthy"])
            self.assertEqual(health["url"], "https://my-bifrost.onrender.com")
            self.assertEqual(health["health_url"], "https://my-bifrost.onrender.com/health")

            # Check that httpx.get was called with the /health endpoint and virtual key header
            mock_get.assert_called_once()
            called_args, called_kwargs = mock_get.call_args
            self.assertEqual(called_args[0], "https://my-bifrost.onrender.com/health")
            self.assertEqual(called_kwargs.get("headers", {}).get("x-bf-vk"), "sk-bf-test-vk")

    def test_chat_completion_automatic_fallback_when_gateway_unreachable(self):
        """When Bifrost is unreachable, chat_completion should automatically fall back to Direct Gemini."""
        mock_gemini_resp = MagicMock()
        mock_gemini_resp.choices = [
            MagicMock(message=MagicMock(content="Fallback answer from direct Gemini"))
        ]
        mock_gemini_resp.model = "gemini-2.5-flash"
        mock_gemini_resp.usage = {"prompt_tokens": 40, "completion_tokens": 15, "total_tokens": 55}
        mock_gemini_resp.model_extra = {}

        # Simulate Bifrost gateway connection failure
        mock_bifrost_client = MagicMock()
        mock_bifrost_client.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock()
        )

        with patch("llm.load_dotenv"), patch.dict(
            os.environ,
            {
                "USE_BIFROST": "true",
                "LLM_API_KEY": "bifrost-local",
                "GEMINI_API_KEY": "AIzaSyTestDirectKey12345",
                "BIFROST_URL": "https://unreachable-gateway.onrender.com",
            },
            clear=False,
        ):
            with patch("llm.get_llm_client", return_value=(mock_bifrost_client, ["gemini/gemini-2.5-flash"], "https://unreachable-gateway.onrender.com/v1")):
                with patch("llm.OpenAI") as mock_openai_cls:
                    mock_direct_gemini_client = MagicMock()
                    mock_direct_gemini_client.chat.completions.create.return_value = mock_gemini_resp
                    mock_openai_cls.return_value = mock_direct_gemini_client

                    content, meta = llm.chat_completion(
                        messages=[{"role": "user", "content": "What is the capital of France?"}],
                    )

                    self.assertEqual(content, "Fallback answer from direct Gemini")
                    self.assertTrue(meta.get("gateway_unreachable_fallback"))
                    self.assertEqual(meta.get("routing_mode"), "direct_gemini_fallback")

    def test_deployment_files_exist_and_follow_security_specifications(self):
        """Verify Dockerfile, config.json, render.yaml, and README.md exist and meet all security rules."""
        config_path = DEPLOY_DIR / "config.json"
        dockerfile_path = DEPLOY_DIR / "Dockerfile"
        render_yaml_path = ROOT_DIR / "render.yaml"
        readme_path = DEPLOY_DIR / "README.md"

        self.assertTrue(config_path.is_file(), "deploy/bifrost/config.json must exist")
        self.assertTrue(dockerfile_path.is_file(), "deploy/bifrost/Dockerfile must exist")
        self.assertTrue(render_yaml_path.is_file(), "root render.yaml must exist")
        self.assertTrue(readme_path.is_file(), "deploy/bifrost/README.md must exist")

        # Parse config.json
        config_data = json.loads(config_path.read_text(encoding="utf-8"))

        # Rule 1: Zero plaintext keys in config.json
        raw_config_text = config_path.read_text(encoding="utf-8")
        from scripts.check_evidence_clean import KEY_PATTERNS
        for name, regex in KEY_PATTERNS:
            self.assertIsNone(
                regex.search(raw_config_text),
                f"Found plaintext secret matching pattern '{name}' in config.json",
            )

        # Rule 2: Providers use env references
        providers = config_data.get("providers", {})
        self.assertIn("gemini", providers)
        gemini_keys = providers["gemini"].get("keys", [])
        self.assertTrue(any(k.get("value") == "env.GEMINI_API_KEY" for k in gemini_keys))

        # Rule 3: Dashboard password protection is enabled
        gov = config_data.get("governance", {})
        auth_cfg = gov.get("auth_config", {})
        self.assertTrue(auth_cfg.get("is_enabled"), "Dashboard authentication must be enabled")
        self.assertEqual(auth_cfg.get("admin_username"), "env.BIFROST_ADMIN_USERNAME")
        self.assertEqual(auth_cfg.get("admin_password"), "env.BIFROST_ADMIN_PASSWORD")

        # Rule 4: Small budget and rate limit configured
        vks = gov.get("virtual_keys", [])
        self.assertTrue(len(vks) > 0, "At least one virtual key must be configured")
        vk = vks[0]
        self.assertEqual(vk.get("value"), "env.BIFROST_VIRTUAL_KEY")
        self.assertTrue(vk.get("is_active"))

        # Budget cap <= $10
        budgets = vk.get("budgets", [])
        self.assertTrue(len(budgets) > 0, "Virtual key must have budget limit")
        self.assertLessEqual(budgets[0].get("max_limit", 999), 10.0, "Budget limit must be small ($5-$10)")

        # Rate limits configured
        rate_limits = gov.get("rate_limits", [])
        self.assertTrue(len(rate_limits) > 0, "Rate limits must be configured")
        self.assertLessEqual(rate_limits[0].get("request_max_limit", 999), 120, "Rate limit should throttle traffic")


if __name__ == "__main__":
    unittest.main()
