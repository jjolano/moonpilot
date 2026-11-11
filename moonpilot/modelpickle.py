import io
import pickle
import shutil
import struct
import tempfile



def dump_oob(obj, handle, scratch_dir) -> None:
  with tempfile.TemporaryFile(dir=scratch_dir) as scratch:
    def write_buffer(buffer: pickle.PickleBuffer):
      view = buffer.raw()
      scratch.write(struct.pack("<q", view.nbytes))
      scratch.write(view)
      buffer.release()

    stream = io.BytesIO()
    pickle.Pickler(stream, protocol=5, buffer_callback=write_buffer).dump(obj)
    opcodes = stream.getvalue()
    handle.write(struct.pack("<q", len(opcodes)))
    handle.write(opcodes)
    scratch.seek(0)
    shutil.copyfileobj(scratch, handle)


def load_oob(handle):
  opcodes = handle.read(struct.unpack("<q", handle.read(8))[0])

  def buffers():
    while chunk := handle.read(8):
      buf = pickle.PickleBuffer(bytearray(struct.unpack("<q", chunk)[0]))
      if handle.readinto(buf) != buf.raw().nbytes:
        raise EOFError("incomplete model buffer")
      yield buf

  return pickle.load(io.BytesIO(opcodes), buffers=buffers())
