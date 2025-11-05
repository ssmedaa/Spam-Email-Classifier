def batcher(items, batch_size=16):

    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
