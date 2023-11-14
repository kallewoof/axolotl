# # class TestEmitter:
# #     def __init__(self, l):
# #         self.l = l
# #     def __iter__(self):
# #         for i in self.l:
# #             yield i

# class TokenizedEmitter:
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer

#     def decode_tokenized(self, data):
#         if "input_ids" in data: data = data["input_ids"]
#         return [("".join(self.tokenizer.convert_ids_to_tokens(d))).replace("▁", " ").replace("<0x0A>", "\n") for d in data]

#     def __call__(self, data):
#         print(self.decode_tokenized(data))

# class EmittingIterator:
#     def __init__(self, iterable, emitter):
#         self._iterable = iterable
#         self._emitter = emitter

#     def __iter__(self):
#         for i in self._iterable:
#             self._emitter(i)
#             yield i
