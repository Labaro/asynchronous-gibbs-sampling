class Problem:
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim

    def split(self, nb_workers):
        return [None for i in range(nb_workers)]