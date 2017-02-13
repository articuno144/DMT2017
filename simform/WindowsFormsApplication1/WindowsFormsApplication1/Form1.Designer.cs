namespace WindowsFormsApplication1
{
    partial class DMT
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(DMT));
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.game_screen = new System.Windows.Forms.Panel();
            this.demo = new System.Windows.Forms.Button();
            this.calibration = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.pictureBox3 = new System.Windows.Forms.PictureBox();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.drone3 = new System.Windows.Forms.PictureBox();
            this.drone1 = new System.Windows.Forms.PictureBox();
            this.drone2 = new System.Windows.Forms.PictureBox();
            this.pictureBox2 = new System.Windows.Forms.PictureBox();
            this.game_screen.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.drone3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.drone1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.drone2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).BeginInit();
            this.SuspendLayout();
            // 
            // timer1
            // 
            this.timer1.Enabled = true;
            this.timer1.Interval = 10;
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // game_screen
            // 
            this.game_screen.BackColor = System.Drawing.Color.White;
            this.game_screen.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.game_screen.Controls.Add(this.drone3);
            this.game_screen.Controls.Add(this.drone1);
            this.game_screen.Controls.Add(this.drone2);
            this.game_screen.Location = new System.Drawing.Point(50, 397);
            this.game_screen.Name = "game_screen";
            this.game_screen.Size = new System.Drawing.Size(698, 536);
            this.game_screen.TabIndex = 7;
            this.game_screen.Paint += new System.Windows.Forms.PaintEventHandler(this.game_screen_Paint);
            // 
            // demo
            // 
            this.demo.Location = new System.Drawing.Point(50, 133);
            this.demo.Name = "demo";
            this.demo.Size = new System.Drawing.Size(157, 56);
            this.demo.TabIndex = 3;
            this.demo.Text = "Run Drone Demo";
            this.demo.UseVisualStyleBackColor = true;
            this.demo.Click += new System.EventHandler(this.button2_Click);
            // 
            // calibration
            // 
            this.calibration.Location = new System.Drawing.Point(294, 133);
            this.calibration.Name = "calibration";
            this.calibration.Size = new System.Drawing.Size(157, 56);
            this.calibration.TabIndex = 2;
            this.calibration.Text = "Calibrate Controls";
            this.calibration.UseVisualStyleBackColor = true;
            this.calibration.Click += new System.EventHandler(this.button1_Click);
            // 
            // label1
            // 
            this.label1.BackColor = System.Drawing.Color.White;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 12.25F);
            this.label1.Location = new System.Drawing.Point(50, 234);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(698, 128);
            this.label1.TabIndex = 8;
            this.label1.Text = resources.GetString("label1.Text");
            this.label1.UseCompatibleTextRendering = true;
            // 
            // pictureBox3
            // 
            this.pictureBox3.Image = global::WindowsFormsApplication1.Properties.Resources.light_on1;
            this.pictureBox3.Location = new System.Drawing.Point(669, 122);
            this.pictureBox3.Name = "pictureBox3";
            this.pictureBox3.Size = new System.Drawing.Size(89, 76);
            this.pictureBox3.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox3.TabIndex = 10;
            this.pictureBox3.TabStop = false;
            this.pictureBox3.Visible = false;
            // 
            // pictureBox1
            // 
            this.pictureBox1.Image = global::WindowsFormsApplication1.Properties.Resources.light_off2;
            this.pictureBox1.Location = new System.Drawing.Point(669, 122);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(89, 76);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox1.TabIndex = 9;
            this.pictureBox1.TabStop = false;
            // 
            // drone3
            // 
            this.drone3.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.drone3.BackColor = System.Drawing.Color.Transparent;
            this.drone3.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.drone3.Image = global::WindowsFormsApplication1.Properties.Resources.drones;
            this.drone3.Location = new System.Drawing.Point(392, 253);
            this.drone3.Margin = new System.Windows.Forms.Padding(0);
            this.drone3.Name = "drone3";
            this.drone3.Size = new System.Drawing.Size(53, 49);
            this.drone3.TabIndex = 2;
            this.drone3.TabStop = false;
            // 
            // drone1
            // 
            this.drone1.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.drone1.Cursor = System.Windows.Forms.Cursors.Cross;
            this.drone1.Image = global::WindowsFormsApplication1.Properties.Resources.drones;
            this.drone1.Location = new System.Drawing.Point(243, 253);
            this.drone1.Margin = new System.Windows.Forms.Padding(0);
            this.drone1.Name = "drone1";
            this.drone1.Size = new System.Drawing.Size(53, 49);
            this.drone1.TabIndex = 1;
            this.drone1.TabStop = false;
            // 
            // drone2
            // 
            this.drone2.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.drone2.Image = global::WindowsFormsApplication1.Properties.Resources.drones;
            this.drone2.Location = new System.Drawing.Point(316, 254);
            this.drone2.Margin = new System.Windows.Forms.Padding(0);
            this.drone2.Name = "drone2";
            this.drone2.Size = new System.Drawing.Size(53, 49);
            this.drone2.TabIndex = 0;
            this.drone2.TabStop = false;
            // 
            // pictureBox2
            // 
            this.pictureBox2.BackgroundImage = global::WindowsFormsApplication1.Properties.Resources.cooltext229557302086950;
            this.pictureBox2.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
            this.pictureBox2.Location = new System.Drawing.Point(50, 12);
            this.pictureBox2.Name = "pictureBox2";
            this.pictureBox2.Size = new System.Drawing.Size(698, 84);
            this.pictureBox2.TabIndex = 5;
            this.pictureBox2.TabStop = false;
            // 
            // DMT
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.AutoValidate = System.Windows.Forms.AutoValidate.EnableAllowFocusChange;
            this.BackColor = System.Drawing.Color.Black;
            this.ClientSize = new System.Drawing.Size(784, 962);
            this.Controls.Add(this.pictureBox3);
            this.Controls.Add(this.pictureBox1);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.game_screen);
            this.Controls.Add(this.pictureBox2);
            this.Controls.Add(this.demo);
            this.Controls.Add(this.calibration);
            this.KeyPreview = true;
            this.Name = "DMT";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "DMT";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.DMT_KeyDown);
            this.KeyUp += new System.Windows.Forms.KeyEventHandler(this.DMT_KeyUp);
            this.game_screen.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.drone3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.drone1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.drone2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox pictureBox2;
        private System.Windows.Forms.Timer timer1;
        private System.Windows.Forms.PictureBox drone1;
        private System.Windows.Forms.PictureBox drone2;
        private System.Windows.Forms.PictureBox drone3;
        private System.Windows.Forms.Panel game_screen;
        private System.Windows.Forms.Button demo;
        private System.Windows.Forms.Button calibration;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.PictureBox pictureBox3;
    }
}

