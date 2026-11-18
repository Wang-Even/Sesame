"""Restricted, vectorized motion Python: no imports, I/O, loops or host execution.

The AST is interpreted against an explicit CUDA math API. Generated programs
cannot access credentials, the simulator, the validator, or acceptance settings.
This is a Sesame execution adapter, not the unpublished MimicAgent quadruped code.
"""
import ast
import math
import operator

OPS = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
       ast.Div:operator.truediv, ast.Pow:operator.pow, ast.Mod:operator.mod,
       ast.BitAnd:operator.and_, ast.BitOr:operator.or_}
COMPARE = {ast.Lt:operator.lt, ast.LtE:operator.le, ast.Gt:operator.gt,
           ast.GtE:operator.ge, ast.Eq:operator.eq, ast.NotEq:operator.ne}
API_NAMES = {"sin","cos","sqrt","abs","exp","tanh","floor","where",
             "stack","tensor","zeros_like","ones_like","copy","amin","amax",
             "feet","ik"}


def parse_program(source):
    if not isinstance(source,str) or len(source)>40000:
        raise ValueError("Motion program must be text, at most 40000 characters")
    tree = ast.parse(source)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError("Define only generate(t, home, m); no module-level code")
    fn = tree.body[0]
    if (fn.name!="generate" or [x.arg for x in fn.args.args]!=["t","home","m"]
        or fn.decorator_list or fn.args.defaults or fn.args.vararg or fn.args.kwarg
        or fn.args.kwonlyargs or fn.args.posonlyargs or fn.returns
        or any(x.annotation for x in fn.args.args)):
        raise ValueError("Signature must be exactly generate(t, home, m)")
    if len(list(ast.walk(tree)))>5000:
        raise ValueError("Motion program exceeds expression budget")
    allowed = (ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Assign,ast.Return,
        ast.Name,ast.Load,ast.Store,ast.Constant,ast.List,ast.Tuple,ast.Dict,
        ast.BinOp,ast.UnaryOp,ast.USub,ast.UAdd,ast.Invert,ast.Call,ast.Attribute,
        ast.keyword,ast.Subscript,ast.Slice,ast.Compare,*OPS,*COMPARE)
    for node in ast.walk(tree):
        if not isinstance(node,allowed):
            raise ValueError(f"Unsupported syntax: {type(node).__name__}; use vectorized expressions")
        if isinstance(node,ast.Attribute):
            if not isinstance(node.value,ast.Name) or node.value.id!="m" or node.attr not in API_NAMES:
                raise ValueError("Only documented m.API calls are allowed")
        if isinstance(node,ast.Call) and not isinstance(node.func,ast.Attribute):
            raise ValueError("Only documented m.API calls are allowed")
        if isinstance(node,ast.Assign):
            if len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):
                raise ValueError("Assign to a new local name; indexed mutation is not supported")
            if node.targets[0].id in ("t","home","m") or node.targets[0].id.startswith("_"):
                raise ValueError("Cannot replace inputs or use private names")
        if isinstance(node,ast.Name) and node.id.startswith("_"):
            raise ValueError("Private names are not supported")
        if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)):
            try: finite=math.isfinite(node.value)
            except OverflowError: finite=False
            if not finite: raise ValueError("Non-finite constant")
        if isinstance(node,ast.Dict) and any(x is None for x in node.keys):
            raise ValueError("Dictionary expansion is not supported")
        if isinstance(node,ast.keyword) and node.arg is None:
            raise ValueError("Keyword expansion is not supported")
    if not fn.body or not isinstance(fn.body[-1],ast.Return) or any(isinstance(n,ast.Return) for n in fn.body[:-1]):
        raise ValueError("Use one final return statement")
    return fn


def run_program(source, t, home, api):
    """Interpret the finite expression program. Tensor arithmetic remains CUDA."""
    fn = parse_program(source)
    values = {"t":t,"home":home}
    def evaluate(n):
        if isinstance(n,ast.Constant): return n.value
        if isinstance(n,ast.Name):
            if n.id not in values: raise ValueError(f"Unknown name: {n.id}")
            return values[n.id]
        if isinstance(n,(ast.List,ast.Tuple)):
            return [evaluate(x) for x in n.elts] if isinstance(n,ast.List) else tuple(evaluate(x) for x in n.elts)
        if isinstance(n,ast.Dict): return {evaluate(k):evaluate(v) for k,v in zip(n.keys,n.values)}
        if isinstance(n,ast.Slice):
            return slice(*(evaluate(x) if x is not None else None for x in (n.lower,n.upper,n.step)))
        if isinstance(n,ast.Subscript): return evaluate(n.value)[evaluate(n.slice)]
        if isinstance(n,ast.BinOp):
            a,b=evaluate(n.left),evaluate(n.right)
            if isinstance(a,(list,tuple,dict,str)) or isinstance(b,(list,tuple,dict,str)):
                raise ValueError("Arithmetic needs scalars/tensors, not Python containers")
            if isinstance(n.op,ast.Pow) and not (isinstance(b,(int,float)) and abs(b)<=8):
                raise ValueError("Exponent must be a scalar in [-8,8] (execution resource bound)")
            return OPS[type(n.op)](a,b)
        if isinstance(n,ast.UnaryOp):
            return {ast.USub:operator.neg,ast.UAdd:operator.pos,ast.Invert:operator.invert}[type(n.op)](evaluate(n.operand))
        if isinstance(n,ast.Compare):
            if len(n.ops)!=1: raise ValueError("Use (a < b) & (b < c), not chained comparisons")
            return COMPARE[type(n.ops[0])](evaluate(n.left),evaluate(n.comparators[0]))
        if isinstance(n,ast.Call):
            return api[n.func.attr](*[evaluate(x) for x in n.args],**{x.arg:evaluate(x.value) for x in n.keywords})
        raise ValueError(f"Unsupported expression: {type(n).__name__}")
    for node in fn.body:
        if isinstance(node,ast.Assign): values[node.targets[0].id]=evaluate(node.value)
        else: return evaluate(node.value)
