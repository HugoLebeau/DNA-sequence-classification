class Trie(object):
    ''' Retrieval tree class. '''
    
    def __init__(self, depth=0, value=''):
        self.children = {}
        self.depth = depth
        self.value = value
    
    def add_key(self, key):
        if len(key) > 0:
            child = self.value+key[0]
            if child not in self.children:
                self.children[child] = Trie(depth=self.depth+1, value=child)
            self.children[child].add_key(key[1:])
    
    def match_rec(self, key, upto, l, i=-1):
        d_upto = 0 if i < 0 or key[i] == self.value[i] else 1
        if upto >= d_upto:
            if self.children == {}: #leaf
                l.append(self.value)
            for child in self.children:
                self.children[child].match_rec(key, upto-d_upto, l, i+1)
    
    def match(self, key, upto):
        l = []
        self.match_rec(key, upto, l, -1)
        return l
    
    def print_dfs(self):
        for child in self.children:
            print(self.children[child].depth, self.children[child].value)
            self.children[child].print_dfs()
