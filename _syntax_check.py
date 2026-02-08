import ast, sys
for f in sys.argv[1:]:
    try:
        ast.parse(open(f).read())
        print(f"{f}: OK")
    except SyntaxError as e:
        print(f"{f}: SYNTAX ERROR line {e.lineno}: {e.msg}")
        sys.exit(1)
