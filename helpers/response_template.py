from starlette.responses import JSONResponse


def success(message: str, data: None):
    content = {
        "message": message,
        "data": data,
        "result": 1
    }
    return JSONResponse(
        status_code=200,
        content=content
    )


def bad_request(message: str, data: None):
    content = {
        "message": message,
        "data": data,
        "result": -1
    }
    return JSONResponse(
        status_code=400,
        content=content
    )
