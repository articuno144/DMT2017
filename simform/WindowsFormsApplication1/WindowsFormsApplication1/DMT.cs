using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Timers;



namespace WindowsFormsApplication1
{



    public partial class DMT : Form
    {
        bool right;
        bool left;
        bool up;
        bool down;
        bool gesture1;
        bool gesture2;
        bool gesture3;
        bool gesture4;
        bool gesture5;
        bool game = false;
        int calibrationstep = 0;

        public void Delay1()
        {
            var t = Task.Run(async delegate
            {
                await Task.Delay(1000);
            });
            t.Wait();
        }



        public DMT()
        {
            InitializeComponent();
        }



        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }

        private void panel1_Paint(object sender, PaintEventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {


            if (calibrationstep == 0)
            {

                this.label1.Text = "You will be asked to carry out a series of gestures 10 times each. Watch the red light and as it flashes carry out the said gesture. Click on the 'Next' to proceed.";
                calibration.Text = "Next";
                calibrationstep = 1;
            }

            else if (calibrationstep == 1)

            {
                this.label1.Text = "Please clench your first when the light flashes.";

                Delay1();
                Delay1();
                Delay1();

                int i = 0;
                while (i < 10)
                {
                    pictureBox3.Visible = true;
                    Delay1();
                    pictureBox3.Visible = false;
                    Delay1();
                    i++;
                }

                this.label1.Text = "Thank you, press 'Next' to move on to the next gesture.";

                calibrationstep = 2;

            }

            else if (calibrationstep == 2)
            {
                this.label1.Text = "Please carry out gesture 2 when the light flashes.";

                Delay1();
                Delay1();
                Delay1();

                int i = 0;
                while (i < 10)
                {
                    pictureBox3.Visible = true;
                    Delay1();
                    pictureBox3.Visible = false;
                    Delay1();
                    i++;
                }

                this.label1.Text = "Thank you, press 'Next' to move on to the next gesture.";

                calibrationstep = 3;

            }

            else if (calibrationstep == 3)

            {
                this.label1.Text = "Please carry out gesture 3 when the light flashes.";

                Delay1();
                Delay1();
                Delay1();

                int i = 0;
                while (i < 10)
                {
                    pictureBox3.Visible = true;
                    Delay1();
                    pictureBox3.Visible = false;
                    Delay1();
                    i++;
                }

                this.label1.Text = "Thank you, press 'Next' to move on to the next gesture.";

                calibrationstep = 4;

            }

            else if (calibrationstep == 4)
            {
                this.label1.Text = "Please carry out gesture 4 when the light flashes.";

                Delay1();
                Delay1();
                Delay1();

                int i = 0;
                while (i < 10)
                {
                    pictureBox3.Visible = true;
                    Delay1();
                    pictureBox3.Visible = false;
                    Delay1();
                    i++;
                }

                this.label1.Text = "Thank you, press 'Next' to move on to the next gesture.";

                calibrationstep = 5;
            }

            else if (calibrationstep == 5)
            {
                this.label1.Text = "Please carry out gesture 5 when the light flashes.";

                Delay1();
                Delay1();
                Delay1();

                int i = 0;
                while (i < 10)
                {
                    pictureBox3.Visible = true;
                    Delay1();
                    pictureBox3.Visible = false;
                    Delay1();
                    i++;
                }

                this.label1.Text = "You have now finished the calibration process. Press next and then you may begin controlling the drone simulation.";

                calibrationstep = 6;
            }

            else if (calibrationstep == 6)
            {
                this.label1.Text = "Hello, welcome to the Drone Swarm Simulation. First click on the 'Calibrate Controls' button to synchronise your hand movements with the drone swarm. After this click on 'Run Drone Demo' and begin controlling the drones with your hand!";
                calibrationstep = 0;
            }




        }

        private void button2_Click(object sender, EventArgs e)
        {

            if (game == false)
            {
                this.label1.Text = "Use WASD to control the Drone and numbers 1-5 for different commands.";
                game = true;
                demo.Text = "Stop Drone Demo";
            }
            else if (game == true)
            {
                this.label1.Text = "Hello, welcome to the Drone Swarm Simulation. First click on the 'Calibrate Controls' button to synchronise your hand movements with the drone swarm. After this click on 'Run Drone Demo' and begin controlling the drones with your hand!";
                game = false;
                demo.Text = "Run Drone Demo";
            }




        }

        private void richTextBox1_TextChanged(object sender, EventArgs e)
        {

        }


        private void timer1_Tick(object sender, EventArgs e)
        {
            if (game == true)
            {
                if (right == true) { drone1.Left += 5; drone2.Left += 5; drone3.Left += 5; }
                if (left == true) { drone1.Left -= 5; drone2.Left -= 5; drone3.Left -= 5; }
                if (up == true) { drone1.Top -= 5; drone2.Top -= 5; drone3.Top -= 5; }
                if (down == true) { drone1.Top += 5; drone2.Top += 5; drone3.Top += 5; }
                if (gesture1 == true) { drone1.Left -= 15; drone3.Left += 15; gesture1 = false; }
                if (gesture2 == true) { drone2.Top -= 15; gesture2 = false; }

            }

        }

        private void DMT_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.D) { right = true; }
            if (e.KeyCode == Keys.A) { left = true; }
            if (e.KeyCode == Keys.W) { up = true; }
            if (e.KeyCode == Keys.S) { down = true; }
            if (e.KeyCode == Keys.D1) { gesture1 = true; }
            if (e.KeyCode == Keys.D2) { gesture2 = true; }
            if (e.KeyCode == Keys.D3) { gesture3 = true; }
            if (e.KeyCode == Keys.D4) { gesture4 = true; }
            if (e.KeyCode == Keys.D5) { gesture5 = true; }
        }

        private void DMT_KeyUp(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.D) { right = false; }
            if (e.KeyCode == Keys.A) { left = false; }
            if (e.KeyCode == Keys.W) { up = false; }
            if (e.KeyCode == Keys.S) { down = false; }
            //if (e.KeyCode == Keys.D1) { gesture1 = false; }
            //if (e.KeyCode == Keys.D2) { gesture2 = false; }
            //if (e.KeyCode == Keys.D3) { gesture3 = false; }
            //if (e.KeyCode == Keys.D4) { gesture4 = false; }
            //if (e.KeyCode == Keys.D5) { gesture5 = false; }

        }

        private void game_screen_Paint(object sender, PaintEventArgs e)
        {

        }
    }
}
