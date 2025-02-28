"""
异常处理模块
定义系统中使用的自定义异常类
"""

class SBSBaseException(Exception):
    """基础异常类"""
    def __init__(self, message: str = None):
        self.message = message or "发生未知错误"
        super().__init__(self.message)

class ConfigError(SBSBaseException):
    """配置错误"""
    pass

class APIError(SBSBaseException):
    """API调用错误"""
    pass

class DataError(SBSBaseException):
    """数据处理错误"""
    pass

class ModelError(SBSBaseException):
    """模型相关错误"""
    pass

class ValidationError(SBSBaseException):
    """数据验证错误"""
    pass

class DatabaseError(SBSBaseException):
    """数据库错误"""
    pass

class NetworkError(SBSBaseException):
    """网络错误"""
    pass

class AuthenticationError(SBSBaseException):
    """认证错误"""
    pass

class AuthorizationError(SBSBaseException):
    """授权错误"""
    pass

class ResourceNotFoundError(SBSBaseException):
    """资源未找到错误"""
    pass

class ResourceExistsError(SBSBaseException):
    """资源已存在错误"""
    pass

class TimeoutError(SBSBaseException):
    """超时错误"""
    pass

class SignalError(SBSBaseException):
    """信号处理错误"""
    pass

class TrainingError(SBSBaseException):
    """训练过程错误"""
    pass 