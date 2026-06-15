class ProjectException(Exception):
    """Base exception for the rifd model
    Args:
        message: str = Human-readable error desc
        error_detail(any, mostly optional) = Additional technical detail"""
    def __init__(self, message:str, error_detail=None):
        super().__init__()
        self.error_detail = error_detail
    def __str__(self):
        base = f"[ProjectException]{super().__str__()}"
        if self.error_detail is not None:
            return f"{base} | Detail: {self.error_detail}"
        return base
    
