"""Render prose blocks with contiguous Markdown table rows."""
def markdown(blocks):
    result=[];previous=''
    for block in blocks:
        if result:result.append('\n' if previous.startswith('|') and block.startswith('|') else '\n\n')
        result.append(block);previous=block
    return ''.join(result)+'\n'
