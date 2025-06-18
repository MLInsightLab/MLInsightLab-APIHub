from pydantic import BaseModel


class PredictRequest(BaseModel):
    model_name: str
    model_flavor: str
    model_version_or_alias: str | int
    data: list
    predict_function: str = 'predict'
    dtype: str = None
    params: dict = None
    convert_to_numpy: bool = True


class LoadRequest(BaseModel):
    model_name: str
    model_flavor: str
    model_version_or_alias: str | int
    requirements: str | None = None
    quantization_kwargs: dict | None = None
    kwargs: dict | None = None


class UserInfo(BaseModel):
    username: str
    role: str
    api_key: str | None = None
    password: str | None = None


class VariableSetRequest(BaseModel):
    variable_name: str
    value: str | int | float | bool | dict | list
    overwrite: bool = False


class VariableDownloadRequest(BaseModel):
    variable_name: str | int | float | bool | dict | list


class VariableDeleteRequest(BaseModel):
    variable_name: str


class VerifyPasswordInfo(BaseModel):
    username: str
    password: str


class VerifyTokenInfo(BaseModel):
    username: str
    token: str
