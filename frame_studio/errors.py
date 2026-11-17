class StudioError(Exception):
    def __init__(self, message, *, code='invalid_request', details=None, status=400):
        super().__init__(message)
        self.message, self.code = message, code
        self.details, self.status = details or [], status
