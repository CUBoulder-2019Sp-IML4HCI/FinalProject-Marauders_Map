from pythonosc import dispatcher
from pythonosc import osc_server
from visualizer import Visualizer
import asyncio

ip = "127.0.0.1"
port = 8999

v = Visualizer()
loop = asyncio.get_event_loop()
task = loop.create_task(v.main_viz_loop())
loop.run_until_complete(task)

def print_volume_handler(unused_addr, args, volume):
  print("[{0}] ~ {1}".format(args[0], volume))

def print_compute_handler(unused_addr, args, volume):
  try:
    print("[{0}] ~ {1}".format(args[0], args[1](volume)))
  except ValueError: pass

def print_faces(unused_addr,args,midX,midY):
    try:
        print(args,midX,midY)
        # for i,arg in enumerate(args):
        #     print(i,args)
        v.drawCircle((midX,midY))
    except ValueError:
        pass
        
if __name__ == "__main__":
  dispatcher = dispatcher.Dispatcher()
  dispatcher.map("/faces", print_faces)
#   dispatcher.map("/volume", print_volume_handler, "Volume")
#   dispatcher.map("/logvolume", print_compute_handler, "Log volume", math.log)

  server = osc_server.ThreadingOSCUDPServer(
      (ip, port), dispatcher)
  print("Serving on {}".format(server.server_address))
  server.serve_forever()
