using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections;
using System.IO;
using System.Diagnostics;

namespace WRCI4111_Ass2_Collier
{
    class Program
    {
        static int p; //pattern p
        static int i; //input layer neuron i
        static int j; //hidden layer neuron j
        static int k; //output layer neuron k
        static double zi; //input signal to neuron i for specific pattern p
        static double yj; //output of hidden neuron j for a specific pattern p
        static double ok; //ouput of output neuron k for a specific pattern p
        static double tk; //target output for a specific pattern p
        static double oFault = 0; //output fault number
        static double tFault = 0; //target fault number
        static double Ep; //Training error per pattern
        static double Et; //Training error
        static double MSE; //Mean squared error
        static double SSE; //Sum squared error
        static double NETyj; //Net input signal to An in hidden layer
        static double NETok; //Net input signal to An in output layer
        static double dEw; //differentiation of error w.r.t. output layer weights
        static double dEv; ////differentiation of error w.r.t. output layer weights
        static double[,] Dt = new double[1800, 28]; //training set of p patterns, 27 signals (I) each
        static double[,] Dv = new double[100, 28]; //validation set of 1900-p patterns, 27 signals (I) each
        static double[,] Dg = new double[40, 28]; //test set of 40 patterns, 27 signals (I) each
        static int P = Dt.GetLength(0); //number of patterns P
        static int I = Dt.GetLength(1) - 1; //number of input layers I
        static int J = 10; //number of hidden layers J
        static int K = 7; //number of output layers K
        static double n = 0.07; //learning rate
        static double t = 0; //epochs initialised to 0
        static double a = 0.5; //scalar value for momentum
        static double[] y = new double[J + 1]; //output of hidden neurons
        static double[] NETy = new double[J]; //input to hidden neurons
        static double[] o = new double[K]; //output of output neurons
        static double[] NETo = new double[K]; //input to output neurons
        static double[] tar = new double[K]; //target values
        static double[,] vji = new double[J, (I + 1)]; //weights between hidden unit and input unit
        static double[,] wkj = new double[K, (J + 1)]; //weights between output unit (only 1) and hidden unit
        static ArrayList SSEvsIterations = new ArrayList(); //used to store SSE vs Iterations training data
        static Stopwatch sw = new Stopwatch();

        static void Main(string[] args)
        {
            vji = InitialiseWeights(vji, 28); //initialise weights v
            wkj = InitialiseWeights(wkj, 11); //initialise weights w

            Read();
            sw.Start();
            Train();
            sw.Stop();
            Validate();
            Test();
            Write();
            Console.WriteLine("Training time:\t" + (sw.ElapsedMilliseconds).ToString());
            Console.ReadLine();
        }

        public static double[,] InitialiseWeights(double[,] weightArr, int fanin)
        {
            //Appropriate weight initialization
            //populate weight array with random floating point number between -1/root(fanin) and 1/root(fanin)
            int rows = weightArr.GetLength(0);
            int columns = weightArr.GetLength(1);
            Random rnd = new Random();
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < columns; c++)
                {
                    //weightArr[r, c] = Scale(rnd.NextDouble(), 1 / Math.Sqrt(fanin), -1 / Math.Sqrt(fanin), 1, 0);
                    weightArr[r, c] = rnd.NextDouble();
                }
            }
            return weightArr;
        }

        public static double[,] NoiseInjection(double[,] trainingArr)
        {
            return trainingArr;
        }

        public static void Write()
        {
            StreamWriter SW = new StreamWriter("SSEvsIterations.csv");
            for (int k = 0; k < SSEvsIterations.Count; k++)
            {
                SW.WriteLine((k + 1).ToString() + "," + (string)SSEvsIterations[k].ToString());
            }
        }

        public static void Read()
        {
            Console.WriteLine("Reading...");
            //Read information from Bin->Debug
            StreamReader SR = new StreamReader("Plates2.csv");
            //Populate training set
            p = 0;
            while (p < P)
            {
                string[] signals = (SR.ReadLine()).Split(',');
                for (i = 0; i < 28; i++)
                {
                    Dt[p, i] = double.Parse(signals[i]); //populate the input signals (zi)
                }
                p++;
            }
            //Populate validation set
            while (p < 1900)
            {
                string[] signals = (SR.ReadLine()).Split(',');
                for (i = 0; i < 28; i++)
                {
                    Dv[p - P, i] = double.Parse(signals[i]); //populate the input signals (zi)
                }
                p++;
            }
            //Populate test set
            while (p < 1940)
            {
                string[] signals = (SR.ReadLine()).Split(',');
                for (i = 0; i < I; i++)
                {
                    Dg[p - 1900, i] = double.Parse(signals[i]); //populate the input signals (zi)
                }
                p++;
            }
            SR.Close();
            //Scale Inputs
            //Dt = ScaleArr(Dt);
            //Dv = ScaleArr(Dv);
            //Dg = ScaleArr(Dg);
        }

        public static void Train()
        {

            double prev_weight_change_v = 0;
            double prev_weight_change_w = 0;
            double prev_weight_change_v_bias = 0;
            double prev_weight_change_w_bias = 0;
            double correct_predictions;
            double accuracy_on_training_set;
            Console.WriteLine("Training...");
            do
            {
                Et = 0;
                correct_predictions = 0;
                for (p = 0; p < P; p++)
                {
                    Ep = 0;
                    //Feedforward Pass
                    //Calculate nety(j) array and scale it
                    for (j = 0; j < J; j++)
                    {
                        NETyj = 0;
                        for (i = 0; i < I; i++)
                        {
                            zi = Dt[p, i];
                            NETyj += vji[j, i] * zi;
                        }
                        zi = -1;
                        NETyj += vji[j, I] * zi;//add bias * v(j, I + 1) to nety(j)
                        NETy[j] = NETyj;
                    }

                    //Calculate y(j) array
                    for (j = 0; j < J; j++)
                    {
                        y[j] = Sigmoid(NETy[j]);
                    }
                    y[J] = -1; //add bias  y(J + 1)

                    //Calculate neto(k) array and scale it
                    for (k = 0; k < K; k++)
                    {
                        NETok = 0;
                        for (j = 0; j < J; j++)
                        {
                            NETok += wkj[k, j] * y[j];
                        }
                        NETok += wkj[k, j] * y[J];
                        NETo[k] = NETok;
                    }

                    //calculate o(k) array and t(k) array for pattern p
                    for (k = 0; k < K; k++)
                    {
                        ok = Sigmoid(NETo[k]);
                        o[k] = ok;
                        if (Dt[p, Dt.GetLength(1) - 1] - 1 == k)
                            tk = 1;
                        else
                            tk = 0;
                        tar[k] = tk;
                        Ep += Math.Pow((tk - ok), 2);
                    }

                    //Backward Propogation
                    //Update w
                    for (k = 0; k < K; k++)
                    {
                        for (j = 0; j < J; j++)
                        {
                            dEw = -(tar[k] - o[k]) * o[k] * (1 - o[k]) * y[j]; //update w.r.t. hidden layer neurons
                            wkj[k, j] = wkj[k,j] - n * dEw + a * prev_weight_change_w;
                            prev_weight_change_w = -n * dEw;
                        }
                        dEw = -(tar[k] - o[k]) * o[k] * (1 - o[k]) * y[J]; //update with bias from hidden layer too
                        wkj[k, J] = wkj[k,j] - n * dEw + a * prev_weight_change_w_bias;
                        prev_weight_change_w_bias = -n * dEw;
                    }
                    //Update v
                    
                    for (j = 0; j < J; j++)
                    {
                        for (i = 0; i < I; i++)
                        {
                            dEv = 0;
                            for (int kk = 0; kk < K; kk++)
                            {
                                zi = Dt[p, i]; //update w.r.t. unput layer neurons
                                dEv -= (tar[kk] - o[kk]) * o[kk] * (1 - o[kk]) * wkj[kk, j] * y[j] * (1 - y[j]) * zi;
                            } 
                            vji[j, i] = vji[j,i] - n * dEv + a * prev_weight_change_v;
                            prev_weight_change_v = -n * dEv; //save weight change for momentum term in following pattern
                        }

                        dEv = 0;
                        for (int kk = 0; kk < K; kk++)
                        {
                            zi = -1; //update with bias from input layer too
                            dEv -= (tar[kk] - o[kk]) * o[kk] * (1 - o[kk]) * wkj[kk, j] * y[j] * (1 - y[j]) * zi;
                        }
                        vji[j, I] = vji[j, I] - n * dEv + a * prev_weight_change_v_bias;
                        prev_weight_change_v_bias = -n * dEv; //save weight change for momentum term in following pattern
                    }

                    //check predictions
                    double error = 0;
                    for (k = 0; k < K; k++)
                    {
                        error += Math.Abs(Math.Round(o[k], 0) - tar[k]);
                    }
                    if (error == 0)
                        correct_predictions++;
                    //Error
                    Et += Ep;
                }
                accuracy_on_training_set = (correct_predictions / Dt.GetLength(0)) * 100;
                SSE = Et;
                SSEvsIterations.Add(SSE); //Record SSE for output
                t++;
                if (t % 5 == 0)
                {
                    Console.WriteLine("Iteration:\t" + t + "\tSSE:\t" + SSE.ToString() + "\tAccuracy:\t" + accuracy_on_training_set.ToString());
                }
            }
            while (t < 5000 || SSE < 500 || accuracy_on_training_set > 80); //run for t iterations
        }

        public static void Validate()
        {
            double correct_predictions = 0;
            double accuracy_on_validation_set;
            Console.WriteLine("Validating...");
            for (p = 0; p < Dv.GetLength(0); p++)
            {
                Ep = 0;
                //Scaling NETyj
                for (j = 0; j < J; j++)
                {
                    NETyj = 0;
                    for (i = 0; i < I; i++)
                    {
                        zi = Dv[p, i];
                        NETyj += vji[j, i] * zi;
                    }
                    zi = -1;
                    NETyj += vji[j, I] * zi;
                    NETy[j] = NETyj;
                }

                for (j = 0; j < J; j++)
                {
                    y[j] = Sigmoid(NETy[j]);
                }
                y[J] = -1;
                //Scaling NETok
                for (k = 0; k < K; k++)
                {
                    NETok = 0;
                    for (j = 0; j < J; j++)
                    {
                        NETok += wkj[k, j] * y[j];
                    }
                    NETok += wkj[k, j] * y[J];
                    NETo[k] = NETok;
                }

                //Calculate tk, ok, tFault, oFault, Ep
                double max = 0;
                for (k = 0; k < K; k++)
                {
                    ok = Sigmoid(NETo[k]);
                    o[k] = ok;
                    tFault = Dv[p, Dv.GetLength(1) - 1];
                    if (tFault - 1 == k)
                        tk = 1;
                    else
                        tk = 0;
                    tar[k] = tk;
                    if (o[k] > max)
                    {
                        oFault = k + 1;
                        max = o[k];
                    }
                    Ep += Math.Pow((tk - ok), 2);
                }
                //check predictions and find fault numbers for traget and output
                double error = 0;
                for (k = 0; k < K; k++)
                {
                    error += Math.Abs(Math.Round(o[k], 0) - tar[k]);
                }
                if (error == 0)
                    correct_predictions++;
                Console.WriteLine("{0}\t\t{1}\t\t{2}", (p + 1).ToString(), tFault.ToString(), oFault.ToString());
                Et += Ep;
            }
            SSE = Et;
            accuracy_on_validation_set = correct_predictions / Dv.GetLength(0) * 100;
            Console.WriteLine("Accuracy:\t" + accuracy_on_validation_set.ToString());
            Console.WriteLine("SSE:\t\t" + SSE.ToString());
        }

        public static void Test()
        {
            Console.WriteLine("Testing...");
            for (p = 0; p < Dg.GetLength(0); p++)
            {
                //Scaling NETyj
                for (j = 0; j < J; j++)
                {
                    NETyj = 0;
                    for (i = 0; i < I; i++)
                    {
                        zi = Dg[p, i];
                        NETyj += vji[j, i] * zi;
                    }
                    zi = -1;
                    NETyj += vji[j, I] * zi;
                    NETy[j] = NETyj;
                }

                for (j = 0; j < J; j++)
                {
                    y[j] = Sigmoid(NETy[j]);
                }
                y[J] = -1;
                //Scaling NETok
                for (k = 0; k < K; k++)
                {
                    NETok = 0;
                    for (j = 0; j < J; j++)
                    {
                        NETok += wkj[k, j] * y[j];
                    }
                    NETok += wkj[k, j] * y[J];
                    NETo[k] = NETok;
                }

                //Calculate tk, ok, tFault, oFault, Ep
                double max = 0;
                for (k = 0; k < K; k++)
                {
                    ok = Sigmoid(NETo[k]);
                    o[k] = ok;
                    if (o[k] > max)
                    {
                        oFault = k + 1;
                        max = o[k];
                    }
                }
                //check predictions and find fault numbers for traget and output
                double error = 0;
                if (error == 0)
                Console.WriteLine("{0}\t\t{1}", (p + 1).ToString(), oFault.ToString());
            }
        }

        public static double Scale(double tu, double tsmax, double tsmin, double tumax, double tumin)
        {
            return ((tu - tumin) / (tumax - tumin)) * (tsmax - tsmin) + (tsmin);
        }

        public static double[,] ScaleArr(double[,] Inputs)
        {
            double tsmin = 0;
            double tsmax = 1;
            for (i = 0; i < Inputs.GetLength(1) - 1; i++)
            {
                double tumin = double.MaxValue;
                double tumax = double.MinValue;
                for (p = 0; p < Inputs.GetLength(0); p++)
                {
                    if (Inputs[p, i] < tumin)
                        tumin = Inputs[p, i];
                    if (Inputs[p, i] > tumax)
                        tumax = Inputs[p, i];
                }
                for (p = 0; p < Inputs.GetLength(1); p++)
                {
                    Inputs[p, i] = Scale(Inputs[p, i], tsmax, tsmin, tumax, tumin);
                }
            }
            return Inputs;
        }

        public static double Sigmoid(double x)
        {
            return 1 / (1  + Math.Exp(-x));
        }

        
    }
}
