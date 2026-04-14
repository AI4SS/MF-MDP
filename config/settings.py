"""
配置加载模块
支持从环境变量、YAML配置文件和命令行参数加载配置
"""
import os
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

from dotenv import load_dotenv


class Config:
    """配置类，支持多级配置加载和覆盖"""

    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置

        Args:
            config_file: YAML配置文件路径，默认为 config/config.yaml
        """
        self._config = {}
        self._project_root = self._find_project_root()

        # 加载环境变量
        load_dotenv()

        # 加载YAML配置文件
        if config_file is None:
            config_file = self._project_root / "config" / "config.yaml"

        if Path(config_file).exists() and yaml is not None:
            with open(config_file, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}

    def _find_project_root(self) -> Path:
        """自动查找项目根目录"""
        current = Path(__file__).parent.parent
        return current

    @property
    def project_root(self) -> Path:
        """项目根目录"""
        return self._project_root

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        获取配置值，优先级：环境变量 > YAML配置 > 默认值

        Args:
            key: 配置键名（支持点号分隔的嵌套键，如 "paths.event_data_dir"）
            default: 默认值

        Returns:
            配置值或默认值
        """
        # 1. 优先从环境变量获取
        env_key = key.upper().replace('.', '_')
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value

        # 2. 从YAML配置获取
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        # 如果是相对路径，转换为绝对路径
        if isinstance(value, str) and value.startswith('./'):
            return str(self._project_root / value[2:])

        return value if value is not None else default

    def get_path(self, key: str, default: Optional[str] = None) -> Path:
        """获取配置路径，返回Path对象"""
        return Path(self.get(key, default) or default)

    @property
    def paths(self):
        """路径配置"""
        class Paths:
            def __init__(self, config):
                self._config = config

            def event_data_dir(self) -> Path:
                return self._config.get_path('paths.event_data_dir', './data/events')

            def mf_data_dir(self) -> Path:
                return self._config.get_path('paths.mf_data_dir', './data/test_mf')

            def state_trajectory_dir(self) -> Path:
                return self._config.get_path('paths.state_trajectory_dir', './data/test_state_distribution')

            def profile_path(self) -> Path:
                return self._config.get_path('paths.profile_path', './main/data/profile/cluster_core_user_profile.jsonl')

            def state_model_checkpoint(self) -> Path:
                return self._config.get_path('paths.state_model_checkpoint', './checkpoints/event_transformer/event_transformer_best.pt')

            def policy_model_path(self) -> Path:
                return self._config.get_path('paths.policy_model_path', './models/qwen2-1.5B')

            def mf_model_path(self) -> Path:
                return self._config.get_path('paths.mf_model_path', './models/qwen2-1.5B')

            def model_base_dir(self) -> Path:
                return self._config.get_path('paths.model_base_dir', './models/')

            def output_dir(self) -> Path:
                return self._config.get_path('paths.output_dir', './main/result')

            def cache_dir(self) -> Path:
                return self._config.get_path('paths.cache_dir', './cache/text_embeddings')

            def pred_state_dir(self) -> Path:
                return self._config.get_path('paths.pred_state_dir', './data/pred_state_distribution')

        return Paths(self)

    @property
    def api(self):
        """API配置"""
        class API:
            def __init__(self, config):
                self._config = config

            def vllm_base_url(self) -> str:
                return self._config.get('api.vllm_base_url', 'http://localhost:8000/v1')

            def vllm_model_name(self) -> str:
                return self._config.get('api.vllm_model_name', 'deepseek-chat')

        return API(self)


# 全局配置实例
config = Config()


def get_config(config_file: Optional[str] = None) -> Config:
    """获取配置实例"""
    return Config(config_file)
