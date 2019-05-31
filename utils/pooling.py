# Author: llw
import torch as t


def global_pooling(input_features):
    max_feature = t.max(input_features, 0).values
    var_feature = t.var(input_features, 0)
    return t.cat([max_feature, var_feature])


if __name__ == '__main__':
    a = t.randn(3, 4);
    print(a)
    print(global_pooling(a))