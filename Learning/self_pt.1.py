def decorator(func):
    def wrapper(self, *args, **kwargs):
        print("Before calling method")
        result = func(self, *args, **kwargs)
        print("After calling method")
        return result
    return wrapper

class Sample:
    @decorator
    def hello(self):
        print("Hello, world!")



if __name__ == "__main__":
    sample = Sample()
    sample.hello()