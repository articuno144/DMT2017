using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.IO.Pipes;

namespace pipe_test_2_pre
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
            var br = new BinaryReader(server);
            var bw = new BinaryWriter(server);
            string str = new string('Message[0]');

            while (true)
            {
                try
                {
                    var buf = Encoding.ASCII.GetBytes(str);     // Get ASCII byte array     
                    bw.Write((uint)buf.Length);                // Write string length
                    bw.Write(buf);                              // Write string
                    Console.WriteLine("Wrote: \"{0}\"", str);

                    var len = (int)br.ReadUInt32();            // Read string length
                    var str = new string(br.ReadChars(len));    // Read string

                    Console.WriteLine("Read: \"{0}\"", str);

                    str = new string(str.Reverse().ToArray());  // Just for fun

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
