"""Evaluate explicit numeric equation subexpressions without eval or guessing."""
import ast
from decimal import Decimal,localcontext
from fractions import Fraction

def equation_values(text,gold=None):
    values=[]
    def visit(n):
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)) and not isinstance(n.value,bool):return Fraction(str(n.value))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            v=visit(n.operand);return -v if isinstance(n.op,ast.USub) else v
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            a,b=visit(n.left),visit(n.right)
            v=a+b if isinstance(n.op,ast.Add) else a-b if isinstance(n.op,ast.Sub) else a*b if isinstance(n.op,ast.Mult) else a/b
            values.append(v);return v
        raise ValueError('Unsupported equation syntax')
    if not text or len(text)>2000:raise ValueError('Missing/oversized equation')
    result=None
    for part in text.replace('×','*').replace('÷','/').split(';'):
        lhs,*rhs=part.strip().split('=')
        result=visit(ast.parse(lhs.strip(),mode='eval').body)
        if rhs and result!=Fraction(rhs[-1].strip()):raise ValueError('Equation equality fails')
    if gold is not None and result!=Fraction(str(gold)):raise ValueError('Equation does not match scalar gold')
    if not values:raise ValueError('No computed quantities')
    output=[]
    for v in values:
        d=v.denominator
        for factor in [2,5]:
            while d%factor==0:d//=factor
        if d!=1:raise ValueError('Nonterminating rational intermediate')
        with localcontext() as context:
            context.prec=80;output.append(format(Decimal(v.numerator)/Decimal(v.denominator),'f'))
    return sorted(set(output))
