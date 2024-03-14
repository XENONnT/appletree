import inspect
import appletree as apt


class TestTransformer(apt.Transformer):
    domain = {"g1", "g2"}
    codomain = {"a", "b"}

    def transform(self, param):
        return {
            "a": param["g1"] * 0.5,
            "b": param["g2"] * 2,
        }

    def inverse_transform(self, param):
        return {
            "g1": param["a"] * 2,
            "g2": param["b"] * 0.5,
        }

    def jacobian(self, param):
        return 1.0

    def print(self):
        return inspect.getsource(self)


def test_transformer():
    apt.clear_cache()
    transformer = TestTransformer()
    tree = transformer(apt.ContextRn220)()
    tree.fitting(iteration=10, batch_size=int(1e4))
