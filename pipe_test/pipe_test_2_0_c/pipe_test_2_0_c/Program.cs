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
            server.WaitForConnection();
            var br = new BinaryReader(server);
            var bw = new BinaryWriter(server);
            char[] charArr = new char[] { (char)1, (char)2, (char)3, (char)4, (char)5 };

            var bufInit = Encoding.ASCII.GetBytes(charArr);     // Get ASCII byte array     
            bw.Write((uint)bufInit.Length);                // Write string length
            bw.Write(bufInit);                              // Write string
            Console.WriteLine(charArr[0]+ charArr[1]);

            while (true)
            {
                try
                {
                    var len = (int)br.ReadUInt32();            // Read string length
                    //var str = new string(br.ReadChars(len));    // Read string
                    charArr = br.ReadChars(len);

                    Console.WriteLine(charArr[1]);

                    //str = new string(str.Reverse().ToArray());  // Just for fun

                    var buf = Encoding.ASCII.GetBytes(charArr);     // Get ASCII byte array     
                    bw.Write((uint)buf.Length);                // Write string length
                    bw.Write(buf);                              // Write string
                    Console.WriteLine(charArr[1]);
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
