import cflib
import cflib.crtp

cflib.crtp.init_drivers()
available = cflib.crtp.scan_interfaces()
for i in available:
    print("Interface with URI [%s] found and name/comment [%s]" % (i[0], i[1]))