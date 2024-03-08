import inspect


def get_function_code(func):
    """Print the code of a function and return code object.

    Parameters
    ----------
    func : function
        The function to extract the code from.
    """

    # Get the function's code object
    func_code = func.__code__

    # Convert the code object to a string
    func_text = inspect.getsource(func)

    # Print the function's body to the console
    print(func_text)


# Test the function
def test_get_function_code():
    def test_func():
        print('This is a test function.')

    get_function_code(test_func)


if __name__ == '__main__':
    test_get_function_code()
