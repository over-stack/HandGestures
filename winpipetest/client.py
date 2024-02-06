import time
import sys
import win32pipe, win32file, pywintypes
import struct


pipeName = "MessageServer"
bufferSize = 200
NUM_TOP_CLOSEST_FACES = 1


def main():
    print("pipe client")
    quit = False

    while not quit:
        try:
            handle = win32file.CreateFile(
                r'\\.\pipe\\' + pipeName,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                0,
                None
            )
            res = win32pipe.SetNamedPipeHandleState(handle, win32pipe.PIPE_READMODE_MESSAGE, None, None)
            if res == 0:
                print(f"SetNamedPipeHandleState return code: {res}")
            while True:
                resp = win32file.ReadFile(handle, bufferSize)
                # data = struct.unpack(f"HddddH" + 'dd??' * NUM_TOP_CLOSEST_FACES, resp[1])
                # a, b, c, d = struct.unpack("HddH", resp[1])
                data = resp[1].decode('utf-8')
                print(data)
                # print(f"message: {a}, {b}, {c}, {d}")
        except pywintypes.error as e:
            if e.args[0] == 2:
                print("no pipe, trying again in a sec")
                time.sleep(1)
            elif e.args[0] == 109:
                print("broken pipe, bye bye")
                quit = True


if __name__ == '__main__':
    main()
