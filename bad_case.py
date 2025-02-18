from tinygrad.tensor import Tensor

# 创建基础 tensors
output = Tensor.zeros(1024, 1, 1)
input1 = Tensor.zeros(1024, 50000, 1)
input2 = Tensor.zeros(1024, 50000, 1)

# 创建常量 tensors
const_0 = Tensor.zeros(1024, 50000, 50000)
const_1 = Tensor.ones(1024, 50000, 50000)
const_minus_1 = Tensor.full((1024, 50000, 1), -1)
const_true = Tensor.ones(1024, 50000, 1, dtype="bool")

# 创建 valid mask
# 由于我们不能直接使用 numpy indexing，我们用数学运算来创建 mask
# 创建坐标网格
y = Tensor.arange(50000).reshape(1, 50000, 1).expand(1024, 50000, 50000)
x = Tensor.arange(50000).reshape(1, 1, 50000).expand(1024, 50000, 50000)

# 创建 mask：x >= 49999 条件
valid_mask = x >= 49999

# WHERE 操作
where_result = valid_mask.where(const_1, const_0)

# REDUCE_AXIS 操作 (对axis=2进行求和)
reduced = where_result.sum(axis=2).reshape(1024, 50000, 1)

# ADD 操作
added = reduced + const_minus_1

# CMPNE 操作 (两次)
cmpne1 = input2 != added
cmpne2 = cmpne1 != const_true

# CAST 到 uchar
casted = cmpne2  # .astype('uint8')

# MUL 操作
multiplied = input1 * casted

# 最终的REDUCE_AXIS (对axis=1进行求和)
final_result = multiplied.sum(axis=1)

# STORE 操作 (将结果存储到output)
output = final_result.reshape(1024, 1, 1)

print(output.numpy())
