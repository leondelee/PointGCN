# Author: llw
import torch as t


def global_pooling(input_features):
    max_feature = t.max(input_features, 1).values
    var_feature = t.var(input_features, 1)
    return t.cat([max_feature, var_feature], dim=1)


if __name__ == '__main__':
    a = t.randn(2, 3, 4);
    print(a)
    print(t.max(a, 1).values)
    # print(t.var(a, 1))