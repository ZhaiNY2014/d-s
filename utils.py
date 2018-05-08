import os


def get_root_path(needslash=True):
    test_dir = os.path.dirname(__file__)
    if needslash is True:
        return test_dir + '/'
    else:
        return test_dir

if __name__ == "__main__":
    print(get_root_path())
    print(get_root_path(False))
