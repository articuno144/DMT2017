using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO.Pipes;
using System.IO;

namespace pipe_test_1_c
{
    class Program
    {
        static void Main(string[] args)
        {
            // Open the named pipe.
            var server = new NamedPipeServerStream("NPtest");

            Console.WriteLine("Waiting for connection...");
            server.WaitForConnection();

            Console.WriteLine("Connected.");
            var sr = new StreamReader(server);
            var sw = new StreamWriter(server);
            var br = new BinaryReader(server);
            var bw = new BinaryWriter(server);

            byte[] init = [1, 2, 3];

            while (true)
            {
                try
                {
                    sw.Write((uint)init.Length);                // Write string length
                    sw.Write(init);                              // Write string
                    Console.WriteLine("Wrote: \"{0}\"", init);

                    var len = (int)br.ReadUInt32();            // Read string length
                    var str = new string(br.(len));    // Read string

                    Console.WriteLine("Read: \"{0}\"", str);

                    //str = new string(str.Reverse().ToArray());  // Just for fun


                }
                catch (EndOfStreamException)
                {
                    break;                    // When client disconnects
                }
            }

            Console.WriteLine("Client disconnected.");
            server.Close();
            server.Dispose();
        }
    }
}
