using System.Diagnostics;

namespace llm.cs;

public class Trainer
{
    public static readonly float PI = (float)Math.PI;

    // ----------------------------------------------------------------------------
    // all the individual layers' forward and backward passes

    public static void EncoderForward(
        Span<float> output,
        Span<int> inp,
        Span<float> wte,
        Span<float> wpe,
        int B, int T, int C)
    {
        for (int b = 0; b < B; b++)
            for (int t = 0; t < T; t++)
        //var _inp = inp.ToArray();
        //var _wte = wte.ToArray();
        //var _wpe = wpe.ToArray();
        //var _output = output.ToArray();
        //Parallel.For(0, B * T, v =>
        {
            //int b = v / T;
            //int t = v % T;
            // seek to the output position in out[b,t,:]
            int out_bt = b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            int wte_ix = ix * C;
            // seek to the position in wpe corresponding to the position
            int wpe_t = t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int i = 0; i < C; i++)
            {
                output[out_bt + i] = wte[wte_ix + i] + wpe[wpe_t + i];
            }
        }//);
        //_output.AsSpan().CopyTo(output);
    }

    public static void EncoderBackward(
        Span<float> dwte,
        Span<float> dwpe,
        Span<float> dout,
        Span<int> inp,
        int B, int T, int C)
    {
        //var _inp = inp.ToArray();
        //var _dwte = dwte.ToArray();
        //var _dwpe = dwpe.ToArray();
        //var _dout = dout.ToArray();
        for (int b = 0; b < B; b++)
            for (int t = 0; t < T; t++)
        //Parallel.For(0, B * T, v =>
        {
            //int b = v / T;
            //int t = v % T;
            int dout_bt = b * T * C + t * C;
            int ix = inp[b * T + t];
            int dwte_ix = ix * C;
            int dwpe_t = t * C;
            for (int i = 0; i < C; i++)
            {
                float d = dout[dout_bt + i];
                dwte[dwte_ix + i] += d;
                dwpe[dwpe_t + i] += d;
            }
        }//);
        //_dwte.AsSpan().CopyTo(dwte);
        //_dwpe.AsSpan().CopyTo(dwpe);
        //LogWriteLine("encoder_backward-dwte" , _dwte);
        //LogWriteLine("encoder_backward-dwpe" , _dwpe);
    }

    public static void LayernormForward(
        Span<float> output,
        Span<float> mean,
        Span<float> rstd,
        Span<float> inp,
        Span<float> weight,
        Span<float> bias,
        int B, int T, int C)
    {
        //Log?.WriteLine("out:");
        float eps = 1e-5f;
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                // seek to the input position inp[b,t,:]
                int x = b * T * C + t * C;
                // calculate the mean
                float m = 0.0f;
                for (int i = 0; i < C; i++)
                {
                    m += inp[x + i];
                }
                m /= C;
                // calculate the variance (without any bias correction)
                float v = 0.0f;
                for (int i = 0; i < C; i++)
                {
                    float xshift = inp[x + i] - m;
                    v += xshift * xshift;
                }
                v /= C;
                // calculate the rstd
                float s = 1.0f / (float)Math.Sqrt(v + eps);
                // seek to the output position in out[b,t,:]
                int out_bt = b * T * C + t * C;
                for (int i = 0; i < C; i++)
                {
                    float n = (s * (inp[x + i] - m)); // normalized output
                    float o = n * weight[i] + bias[i]; // scale and shift it
                    output[out_bt + i] = o; // write
                    //Log?.WriteLine(o);
                }
                // cache the mean and rstd for the backward pass later
                mean[b * T + t] = m;
                rstd[b * T + t] = s;
            }
        }
        //LogWriteLine("layernorm_forward-mean" , mean.ToArray());
        //LogWriteLine("layernorm_forward-rstd" , rstd.ToArray());
    }

    public static void LayernormBackward(
        Span<float> dinp,
        Span<float> dweight,
        Span<float> dbias,
        Span<float> dout,
        Span<float> inp,
        Span<float> weight,
        Span<float> mean,
        Span<float> rstd,
        int B, int T, int C)
    {
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                int dout_bt = b * T * C + t * C;
                int inp_bt = b * T * C + t * C;
                int dinp_bt = b * T * C + t * C;
                float mean_bt = mean[b * T + t];
                float rstd_bt = rstd[b * T + t];

                // first: two reduce operations
                float dnorm_mean = 0.0f;
                float dnorm_norm_mean = 0.0f;
                for (int i = 0; i < C; i++)
                {
                    float norm_bti = (inp[inp_bt + i] - mean_bt) * rstd_bt;
                    float dnorm_i = weight[i] * dout[dout_bt + i];
                    dnorm_mean += dnorm_i;
                    dnorm_norm_mean += dnorm_i * norm_bti;
                }
                dnorm_mean /= C;
                dnorm_norm_mean /= C;

                // now iterate again and accumulate all the gradients
                for (int i = 0; i < C; i++)
                {
                    float norm_bti = (inp[inp_bt + i] - mean_bt) * rstd_bt;
                    float dnorm_i = weight[i] * dout[dout_bt + i];
                    // gradient contribution to bias
                    dbias[i] += dout[dout_bt + i];
                    // gradient contribution to weight
                    dweight[i] += norm_bti * dout[dout_bt + i];
                    // gradient contribution to input
                    float dval = 0.0f;
                    dval += dnorm_i; // term 1
                    dval -= dnorm_mean; // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd_bt; // final scale
                    dinp[dinp_bt + i] += dval;
                }
            }
        }
    }

    public static void MatmulForward(
        Span<float> output,
        Span<float> input,
        Span<float> weight,
        Span<float> bias,
        int B, int T, int C, int OC)
    {
        // most of the running time is spent here and in matmul_backward
        // OC is short for "output channels"
        // inp is (B,T,C), weight is (OC, C), bias is (OC)
        // out will be (B,T,OC)
        for (int b = 0; b < B; b++)
            for (int t = 0; t < T; t++)
        //var _bias = bias.Length > 0 ? bias.ToArray() : null;
        //var _input = input.ToArray();
        //var _weight = weight.ToArray();
        //var _output = output.ToArray();
        //Parallel.For(0, B * T, v =>
        {
            //int b = v / T;
            //int t = v % T;
            int out_bt = b * T * OC + t * OC;
            int inp_bt = b * T * C + t * C;
            for (int o = 0; o < OC; o++)
            {
                float val = (bias != null) ? bias[o] : 0.0f;
                int wrow = o * C;
                for (int i = 0; i < C; i++)
                {
                    val += input[inp_bt + i] * weight[wrow + i];
                }
                output[out_bt + o] = val;
            }
        }//);
        //_output.AsSpan().CopyTo(output);
    }

    public static void MatmulBackward(
        Span<float> dinp,
        Span<float> dweight,
        Span<float> dbias,
        Span<float> dout,
        Span<float> inp,
        Span<float> weight,
        int B, int T, int C, int OC)
    {
        // most of the running time is spent here and in matmul_forward
        // this backward could be done in a single "round" of loops
        // but that doesn't afford an efficient parallelization strategy

        // backward into inp first, parallelize over B,T
        for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
        //var _inp = inp.ToArray();
        //var _dbias = dbias.Length > 0 ? dbias.ToArray() : null;
        //var _dout = dout.ToArray();
        //var _dinp = dinp.ToArray();
        //var _weight = weight.ToArray();
        //var _dweight = dweight.ToArray();
        //Parallel.For(0, B * T, v =>
        {
            //int b = v / T;
            //int t = v % T;
            int dout_bt = b * T * OC + t * OC;
            int dinp_bt = b * T * C + t * C;
            for (int o = 0; o < OC; o++)
            {
                int wrow = o * C;
                float d = dout[dout_bt + o];
                for (int i = 0; i < C; i++)
                {
                    dinp[dinp_bt + i] += weight[wrow + i] * d;
                }
            }
        }//);
        // backward into weight/bias, parallelize over output channels OC
        //Parallel.For(0, OC, o =>
        for (int o = 0; o < OC; o++)
        {
            for (int b = 0; b < B; b++)
            {
                for (int t = 0; t < T; t++)
                {
                    int dout_bt = b * T * OC + t * OC;
                    int inp_bt = b * T * C + t * C;
                    int dwrow = o * C;
                    float d = dout[dout_bt + o];
                    if (dbias != null) { dbias[o] += d; }
                    for (int i = 0; i < C; i++)
                    {
                        dweight[dwrow + i] += inp[inp_bt + i] * d;
                    }
                }
            }
        }//);

        //_dinp.AsSpan().CopyTo(dinp);
        //_dweight.AsSpan().CopyTo(dweight);
    }

    public static void AttentionForward(
        Span<float> output,
        Span<float> preatt,
        Span<float> att,
        Span<float> inp,
        int B, int T, int C, int NH)
    {
        // input is (B, T, 3C) Q,K,V
        // preatt, att are (B, NH, T, T)
        // output is (B, T, C)
        int C3 = C * 3;
        int hs = C / NH; // head size
        float scale = 1.0f / (float)Math.Sqrt(hs);
        for (int b = 0; b < B; b++)
            for (int t = 0; t < T; t++)
            //Parallel.For(0, B * T, v =>
            {
                //int b = v / T;
                //int t = v % T;
                for (int h = 0; h < NH; h++)
                {
                    int query_t = +b * T * C3 + t * C3 + h * hs;
                    int preatt_bth = +b * NH * T * T + h * T * T + t * T;
                    int att_bth = +b * NH * T * T + h * T * T + t * T;

                    // pass 1: calculate query dot key and maxval
                    float maxval = -10000.0f; // TODO something better
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        int key_t2 = +b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                        // (query_t) dot (key_t2)
                        float val = 0.0f;
                        for (int i = 0; i < hs; i++)
                        {
                            val += inp[query_t + i] * inp[key_t2 + i];
                        }
                        val *= scale;
                        if (val > maxval)
                        {
                            maxval = val;
                        }

                        preatt[preatt_bth + t2] = val;
                    }

                    // pass 2: calculate the exp and keep track of sum
                    float expsum = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        float expv = (float)Math.Exp(preatt[preatt_bth + t2] - maxval);
                        expsum += expv;
                        att[att_bth + t2] = expv;
                    }
                    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                    // pass 3: normalize to get the softmax
                    for (int t2 = 0; t2 < T; t2++)
                    {
                        if (t2 <= t)
                        {
                            att[att_bth + t2] *= expsum_inv;
                        }
                        else
                        {
                            // causal attention mask. not strictly necessary to set to zero here
                            // only doing this explicitly for debugging and checking to PyTorch
                            att[att_bth + t2] = 0.0f;
                        }
                    }

                    // pass 4: accumulate weighted values into the output of attention
                    int out_bth = +b * T * C + t * C + h * hs;
                    for (int i = 0; i < hs; i++) { output[out_bth + i] = 0.0f; }
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        int value_t2 = +b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
                        float att_btht2 = att[att_bth + t2];
                        for (int i = 0; i < hs; i++)
                        {
                            output[out_bth + i] += att_btht2 * inp[value_t2 + i];
                        }
                    }
                }
            }//);
    }

    public static void AttentionBackward(
        Span<float> dinp,
        Span<float> dpreatt,
        Span<float> datt,
        Span<float> dout,
        Span<float> inp,
        Span<float> att,
        int B, int T, int C, int NH)
    {
        // inp/dinp are (B, T, 3C) Q,K,V
        // att/datt/dpreatt are (B, NH, T, T)
        // dout is (B, T, C)
        int C3 = C * 3;
        int hs = C / NH; // head size
        float scale = 1.0f / (float)Math.Sqrt(hs);

        for (int b = 0; b < B; b++)
            for (int t = 0; t < T; t++)
            {
                for (int h = 0; h < NH; h++)
                {
                    int att_bth = +b * NH * T * T + h * T * T + t * T;
                    int datt_bth = +b * NH * T * T + h * T * T + t * T;
                    int dpreatt_bth = +b * NH * T * T + h * T * T + t * T;
                    int dquery_t = +b * T * C3 + t * C3 + h * hs;
                    int query_t = +b * T * C3 + t * C3 + h * hs;

                    // backward pass 4, through the value accumulation
                    int dout_bth = +b * T * C + t * C + h * hs;
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        int value_t2 = b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
                        int dvalue_t2 = b * T * C3 + t2 * C3 + h * hs + C * 2;
                        for (int i = 0; i < hs; i++)
                        {
                            // in the forward pass this was:
                            // out_bth[i] += att_bth[t2] * value_t2[i];
                            // so now we have:
                            att[datt_bth + t2] += inp[value_t2 + i] * dout[dout_bth + i];
                            dinp[dvalue_t2 + i] += att[att_bth + t2] * dout[dout_bth + i];
                        }
                    }

                    // backward pass 2  3, the softmax
                    // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        for (int t3 = 0; t3 <= t; t3++)
                        {
                            float indicator = t2 == t3 ? 1.0f : 0.0f;
                            float local_derivative = att[att_bth + t2] * (indicator - att[att_bth + t3]);
                            dpreatt[dpreatt_bth + t3] += local_derivative * datt[datt_bth + t2];
                        }
                    }

                    // backward pass 1, the query @ key matmul
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        int key_t2 = b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                        int dkey_t2 = b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                        for (int i = 0; i < hs; i++)
                        {
                            // in the forward pass this was:
                            // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                            // so now we have:
                            dinp[dquery_t + i] += inp[key_t2 + i] * dpreatt[dpreatt_bth + t2] * scale;
                            dinp[dkey_t2 + i] += inp[query_t + i] * dpreatt[dpreatt_bth + t2] * scale;
                        }
                    }
                }
            }

    }

    public static void GeluForward(Span<float> output, Span<float> input, int N)
    {
        var s = (float)Math.Sqrt(2.0f / PI);
        for (int i = 0; i < N; i++)
        {
            var x = input[i];
            var cube = 0.044715f * x * x * x;
            output[i] = 0.5f * x * (1.0f + (float)Math.Tanh(s * (x + cube)));
        }
    }

    public static void GeluBackward(Span<float> dinp, Span<float> inp, Span<float> dout, int N)
    {
        float s = (float)Math.Sqrt(2.0f / PI);
        for (int i = 0; i < N; i++)
        {
            float x = inp[i];
            float cube = 0.044715f * x * x * x;
            float tanh_arg = s * (x + cube);
            float tanh_out = (float)Math.Tanh(tanh_arg);
            float coshf_out = (float)Math.Cosh(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * s * (1.0f + 3.0f * 0.044715f * x * x);
            dinp[i] += local_grad * dout[i];
        }
    }

    public static void ResidualForward(Span<float> output, Span<float> inp1, Span<float> inp2, int N)
    {
        for (int i = 0; i < N; i++)
        {
            output[i] = inp1[i] + inp2[i];
        }
    }

    public static void ResidualBackward(Span<float> dinp1, Span<float> dinp2, Span<float> dout, int N)
    {
        for (int i = 0; i < N; i++)
        {
            dinp1[i] += dout[i];
            dinp2[i] += dout[i];
        }
    }

    public static void SoftmaxForward(Span<float> probs, Span<float> logits, int B, int T, int V)
    {
        // output: probs are (B,T,V) of the probabilities
        // input: logits is (B,T,V) of the unnormalized log probabilities
        for (int b = 0; b < B; b++)
            for (int t = 0; t < T; t++)
        //var _logits = logits.ToArray();
        //var _probs = probs.ToArray();
        //Parallel.For(0, B * T, v =>
        {
            //int b = v / T;
            //int t = v % T;
            // probs <- softmax(logits)
            int logits_bt = +b * T * V + t * V;
            int probs_bt = +b * T * V + t * V;

            float maxval = -10000.0f; // TODO something better
            for (int i = 0; i < V; i++)
            {
                if (logits[logits_bt + i] > maxval)
                {
                    maxval = logits[logits_bt + i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++)
            {
                probs[probs_bt + i] = (float)Math.Exp(logits[logits_bt + i] - maxval);
                sum += probs[probs_bt + i];
            }
            for (int i = 0; i < V; i++)
            {
                probs[probs_bt + i] /= sum;
            }
        }//);
        //_probs.AsSpan().CopyTo(probs);
        //_logits.AsSpan().CopyTo(logits);
    }

    public static void CrossentropyForward(
        Span<float> losses,
        Span<float> probs, 
        Span<int> targets,
        int B, int T, int V)
    {
        // output: losses is (B,T) of the individual losses at each position
        // input: probs are (B,T,V) of the probabilities
        // input: targets is (B,T) of integers giving the correct index in logits
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                // loss = -log(probs[target])
                int probs_bt = +b * T * V + t * V;
                int ix = targets[b * T + t];
                losses[b * T + t] = -(float)Math.Log(probs[probs_bt + ix]);
            }
        }
    }

    public static void CrossentropySoftmaxBackward(
        Span<float> dlogits,
        Span<float> dlosses,
        Span<float> probs,
        Span<int> targets,
        int B, int T, int V)
    {
        // backwards through both softmax and crossentropy
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                int dlogits_bt = +b * T * V + t * V;
                int probs_bt = +b * T * V + t * V;
                float dloss = dlosses[b * T + t];
                int ix = targets[b * T + t];
                for (int i = 0; i < V; i++)
                {
                    float p = probs[probs_bt + i];
                    float indicator = i == ix ? 1.0f : 0.0f;
                    dlogits[dlogits_bt + i] += (p - indicator) * dloss;
                }
            }
        }
    }

    // allocate memory for the parameters and point the individual tensors to the right places
    public static float[] MallocAndPointParameters(Span<int> param_sizes)
    {
        var num_parameters = 0;
        for (var i = 0; i < GPT2.NUM_PARAMETER_TENSORS; i++)
        {
            num_parameters += param_sizes[i];
        }
        return new float[num_parameters];
    }


    public static float[] MallocAndPointActivations(Span<int> act_sizes)
    {
        var num_activations = 0;
        for (var i = 0; i < GPT2.NUM_ACTIVATION_TENSORS; i++)
        {
            num_activations += act_sizes[i];
        }
        return new float[num_activations];
    }

    public static GPT2 GPT2BuildFromCheckpoint( string model_path, int magic = 20240326)
    {
        var model = new GPT2();
        // read in model from a checkpoint file
        if (!File.Exists(model_path))
        {
            Console.WriteLine("Error opening model file\n");
            return null;
        }
        using var fs = new FileStream(model_path, FileMode.Open);
        using var model_file = new BinaryReader(fs);

        var model_header = new int[256];
        for (int i = 0; i < model_header.Length; i++)
        {
            model_header[i] = model_file.ReadInt32();
        }
        if (model_header[0] != magic)
        {
            Console.WriteLine("Bad magic model file");
            return null;
        }
        if (model_header[1] != 1)
        {
            Console.WriteLine("Bad version in model file");
            return null;
        }


        // read in hyperparameters
        int maxT, V, L, NH, C;
        model.Config.MaxSeqLen = maxT = model_header[2];
        model.Config.VocabSize = V = model_header[3];
        model.Config.NumLayers = L = model_header[4];
        model.Config.NumHeads = NH = model_header[5];
        model.Config.Channels = C = model_header[6];
        Console.WriteLine("[GPT-2]");
        Console.WriteLine("max_seq_len: {0}", maxT);
        Console.WriteLine("vocab_size: {0}", V);
        Console.WriteLine("num_layers: {0}", L);
        Console.WriteLine("num_heads: {0}", NH);
        Console.WriteLine("channels: {0}", C);

        // allocate space for all the parameters and read them in
        model.ParamSizes[0] = V * C;
        model.ParamSizes[1] = maxT * C;
        model.ParamSizes[2] = L * C;
        model.ParamSizes[3] = L * C;
        model.ParamSizes[4] = L * (3 * C) * C;
        model.ParamSizes[5] = L * (3 * C);
        model.ParamSizes[6] = L * C * C;
        model.ParamSizes[7] = L * C;
        model.ParamSizes[8] = L * C;
        model.ParamSizes[9] = L * C;
        model.ParamSizes[10] = L * (4 * C) * C;
        model.ParamSizes[11] = L * (4 * C);
        model.ParamSizes[12] = L * C * (4 * C);
        model.ParamSizes[13] = L * C;
        model.ParamSizes[14] = C;
        model.ParamSizes[15] = C;

        var p = 0;
        var pos = new List<int> { 0 };
        for (var i = 0; i < model.ParamSizes.Length; i++)
            pos.Add(p += model.ParamSizes[i]);

        model.Params.Wte = (pos[0], pos[1]);
        model.Params.Wpe = (pos[1], pos[2]);
        model.Params.Ln1w = (pos[2], pos[3]);
        model.Params.Ln1b = (pos[3], pos[4]);
        model.Params.Qkvw = (pos[4], pos[5]);
        model.Params.Qkvb = (pos[5], pos[6]);
        model.Params.AttProjw = (pos[6], pos[7]);
        model.Params.AttProjb = (pos[7], pos[8]);
        model.Params.Ln2w = (pos[8], pos[9]);
        model.Params.Ln2b = (pos[9], pos[10]);
        model.Params.Fcw = (pos[10], pos[11]);
        model.Params.Fcb = (pos[11], pos[12]);
        model.Params.FcProjw = (pos[12], pos[13]);
        model.Params.FcProjb = (pos[13], pos[14]);
        model.Params.Lnfw = (pos[14], pos[15]);
        model.Params.Lnfb = (pos[15], pos[16]);

        // cound the number of paramaters
        var num_parameters = 0;
        for (var i = 0; i < GPT2.NUM_PARAMETER_TENSORS; i++)
        {
            num_parameters += model.ParamSizes[i];
        }
        Console.WriteLine("num_parameters: {0}", num_parameters);
        model.NumParameters = num_parameters;

        // read in all the parameters from file
        model.ParamMemory = MallocAndPointParameters(model.ParamSizes);
        for (int i = 0; i < num_parameters; i++)
        {
            model.ParamMemory[i] = model_file.ReadSingle();
        }
        // other inits
        model.ActsMemory = null;
        model.GradsMemory = null;
        model.M_Memory = null;
        model.V_Memory = null;
        model.GradsActsMemory = null;
        model.Inputs = null;
        model.Targets = null;
        model.BatchSize = 0;
        model.SeqLen = 0;
        model.MeanLoss = -1.0f; // -1.0f will designate no loss

        return model;
    }

    public static bool GPT2Forward(GPT2 model, int[] inputs, int[] targets, int B, int T)
    {
        // targets are optional and could be null

        // ensure the model was initialized or error out
        if (model.ParamMemory == null)
        {
            Console.WriteLine("Error: model was not initialized properly.\n");
            return false;
        }

        // convenience parameters
        int V = model.Config.VocabSize;
        int L = model.Config.NumLayers;
        int NH = model.Config.NumHeads;
        int C = model.Config.Channels;

        // allocate space for all the activations if needed (done here, lazily)
        if (model.ActsMemory == null)
        {
            // record the current B,T as well
            model.BatchSize = B;
            model.SeqLen = T;
            model.ActSizes[0] = B * T * C;
            model.ActSizes[1] = L * B * T * C;
            model.ActSizes[2] = L * B * T;
            model.ActSizes[3] = L * B * T;
            model.ActSizes[4] = L * B * T * 3 * C;
            model.ActSizes[5] = L * B * T * C;
            model.ActSizes[6] = L * B * NH * T * T;
            model.ActSizes[7] = L * B * NH * T * T;
            model.ActSizes[8] = L * B * T * C;
            model.ActSizes[9] = L * B * T * C;
            model.ActSizes[10] = L * B * T * C;
            model.ActSizes[11] = L * B * T;
            model.ActSizes[12] = L * B * T;
            model.ActSizes[13] = L * B * T * 4 * C;
            model.ActSizes[14] = L * B * T * 4 * C;
            model.ActSizes[15] = L * B * T * C;
            model.ActSizes[16] = L * B * T * C;
            model.ActSizes[17] = B * T * C;
            model.ActSizes[18] = B * T;
            model.ActSizes[19] = B * T;
            model.ActSizes[20] = B * T * V;
            model.ActSizes[21] = B * T * V;
            model.ActSizes[22] = B * T;

            int p;
            List<int> pos = [p = 0];
            for (var i = 0; i < model.ActSizes.Length; i++)
                pos.Add(p += model.ActSizes[i]);

            // and now allocate the space
            model.Acts.Encoded = (pos[0], pos[1]);
            model.Acts.Ln1 = (pos[1], pos[2]);
            model.Acts.Ln1Mean = (pos[2], pos[3]);
            model.Acts.Ln1Rstd = (pos[3], pos[4]);
            model.Acts.Qkv = (pos[4], pos[5]);
            model.Acts.Atty = (pos[5], pos[6]);
            model.Acts.Preatt = (pos[6], pos[7]);
            model.Acts.Att = (pos[7], pos[8]);
            model.Acts.AttProj = (pos[8], pos[9]);
            model.Acts.Residual2 = (pos[9], pos[10]);
            model.Acts.Ln2 = (pos[10], pos[11]);
            model.Acts.Ln2Mean = (pos[11], pos[12]);
            model.Acts.Ln2Rstd = (pos[12], pos[13]);
            model.Acts.Fch = (pos[13], pos[14]);
            model.Acts.FchGelu = (pos[14], pos[15]);
            model.Acts.FcProj = (pos[15], pos[16]);
            model.Acts.Residual3 = (pos[16], pos[17]);
            model.Acts.Lnf = (pos[17], pos[18]);
            model.Acts.LnfMean = (pos[18], pos[19]);
            model.Acts.LnfRstd = (pos[19], pos[20]);
            model.Acts.Logits = (pos[20], pos[21]);
            model.Acts.Probs = (pos[21], pos[22]);
            model.Acts.Losses = (pos[22], pos[23]);
            int num_activations = 0;
            for (long i = 0; i < GPT2.NUM_ACTIVATION_TENSORS; i++)
            {
                num_activations += model.ActSizes[i];
            }
            Console.WriteLine("num_activations: {0}", num_activations);
            model.NumActivations = num_activations;
            model.ActsMemory = MallocAndPointActivations(model.ActSizes);

            // also create memory for caching inputs and targets
            model.Inputs = new int[B * T];
            model.Targets = new int[B * T]; // might be unused if we never have targets but it's small
        }
        else
        {
            // validate B,T is no larger than what was previously allocated
            // in principle, we could re-allocate a larger chunk of memory, for now we just error out
            if (B > model.BatchSize || T > model.SeqLen)
            {
                Console.WriteLine("Error: batch size or sequence length is inadequately large");
                Console.WriteLine("Model: B={0} T={1}, Desired: B={2} T={3}", model.BatchSize, model.SeqLen, B, T);
                return false;
            }
        }

        // cache the inputs/targets
        Array.Copy(inputs, model.Inputs, B * T);
        //memcpy(model.inputs, inputs, B * T * sizeof(int));
        if (targets != null)
        {
            Array.Copy(targets, model.Targets, B * T);
            //memcpy(model.targets, targets, B * T * sizeof(int));
        }

        // forward pass
        ParameterTensors param = model.Params; // for brevity
        ActivationTensors acts = model.Acts;
        int acts_residual;
        EncoderForward(
            model.ActsMemory.AsSpan()[acts.Encoded.Item1..acts.Encoded.Item2],
            inputs,
            model.ParamMemory.AsSpan()[param.Wte.Item1..param.Wte.Item2],
            model.ParamMemory.AsSpan()[param.Wpe.Item1..param.Wpe.Item2],
            B, T, C); // encoding goes into residual[0]

        for (int l = 0; l < L; l++)
        {
            acts_residual = l == 0 ? acts.Encoded.Item1 : acts.Residual3.Item1 + ((l - 1) * B * T * C);

            // get the pointers of the weights for this layer
            int param_l_ln1w = param.Ln1w.Item1 + (l * C);
            int param_l_ln1b = param.Ln1b.Item1 + (l * C);
            int param_l_qkvw = param.Qkvw.Item1 + (l * 3 * C * C);
            int param_l_qkvb = param.Qkvb.Item1 + (l * 3 * C);
            int param_l_attprojw = param.AttProjw.Item1 + (l * C * C);
            int param_l_attprojb = param.AttProjb.Item1 + (l * C);
            int param_l_ln2w = param.Ln2w.Item1 + (l * C);
            int param_l_ln2b = param.Ln2b.Item1 + (l * C);
            int param_l_fcw = param.Fcw.Item1 + (l * 4 * C * C);
            int param_l_fcb = param.Fcb.Item1 + (l * 4 * C);
            int param_l_fcprojw = param.FcProjw.Item1 + (l * C * 4 * C);
            int param_l_fcprojb = param.FcProjb.Item1 + (l * C);


            // get the pointers of the activations for this layer
            int acts_l_ln1 = acts.Ln1.Item1 + (l * B * T * C);
            int acts_l_ln1_mean = acts.Ln1Mean.Item1 + (l * B * T);
            int acts_l_ln1_rstd = acts.Ln1Rstd.Item1 + (l * B * T);
            int acts_l_qkv = acts.Qkv.Item1 + (l * B * T * 3 * C);
            int acts_l_atty = acts.Atty.Item1 + (l * B * T * C);
            int acts_l_preatt = acts.Preatt.Item1 + (l * B * NH * T * T);
            int acts_l_att = acts.Att.Item1 + (l * B * NH * T * T);
            int acts_l_attproj = acts.AttProj.Item1 + (l * B * T * C);
            int acts_l_residual2 = acts.Residual2.Item1 + (l * B * T * C);
            int acts_l_ln2 = acts.Ln2.Item1 + (l * B * T * C);
            int acts_l_ln2_mean = acts.Ln2Mean.Item1 + (l * B * T);
            int acts_l_ln2_rstd = acts.Ln2Rstd.Item1 + (l * B * T);
            int acts_l_fch = acts.Fch.Item1 + (l * B * T * 4 * C);
            int acts_l_fch_gelu = acts.FchGelu.Item1 + (l * B * T * 4 * C);
            int acts_l_fcproj = acts.FcProj.Item1 + (l * B * T * C);
            int acts_l_residual3 = acts.Residual3.Item1 + (l * B * T * C);

            // now do the forward pass
            LayernormForward(
                model.ActsMemory.AsSpan()[acts_l_ln1..],
                model.ActsMemory.AsSpan()[acts_l_ln1_mean..],
                model.ActsMemory.AsSpan()[acts_l_ln1_rstd..],
                model.ActsMemory.AsSpan()[acts_residual..],
                model.ParamMemory.AsSpan()[param_l_ln1w..],
                model.ParamMemory.AsSpan()[param_l_ln1b..],
                B, T, C);

            MatmulForward(
                model.ActsMemory.AsSpan()[acts_l_qkv..],
                model.ActsMemory.AsSpan()[acts_l_ln1..],
                model.ParamMemory.AsSpan()[param_l_qkvw..],
                model.ParamMemory.AsSpan()[param_l_qkvb..],
                B, T, C, 3 * C);
            AttentionForward(
                model.ActsMemory.AsSpan()[acts_l_atty..],
                model.ActsMemory.AsSpan()[acts_l_preatt..],
                model.ActsMemory.AsSpan()[acts_l_att..],
                model.ActsMemory.AsSpan()[acts_l_qkv..],
                B, T, C, NH);
            MatmulForward(
                model.ActsMemory.AsSpan()[acts_l_attproj..],
                model.ActsMemory.AsSpan()[acts_l_atty..],
                model.ParamMemory.AsSpan()[param_l_attprojw..],
                model.ParamMemory.AsSpan()[param_l_attprojb..],
                B, T, C, C);
            ResidualForward(
                model.ActsMemory.AsSpan()[acts_l_residual2..],
                model.ActsMemory.AsSpan()[acts_residual..],
                model.ActsMemory.AsSpan()[acts_l_attproj..],
                B * T * C);
            LayernormForward(
                model.ActsMemory.AsSpan()[acts_l_ln2..],
                model.ActsMemory.AsSpan()[acts_l_ln2_mean..],
                model.ActsMemory.AsSpan()[acts_l_ln2_rstd..],
                model.ActsMemory.AsSpan()[acts_l_residual2..],
                model.ParamMemory.AsSpan()[param_l_ln2w..],
                model.ParamMemory.AsSpan()[param_l_ln2b..],
                B, T, C);
            MatmulForward(
                model.ActsMemory.AsSpan()[acts_l_fch..],
                model.ActsMemory.AsSpan()[acts_l_ln2..],
                model.ParamMemory.AsSpan()[param_l_fcw..],
                model.ParamMemory.AsSpan()[param_l_fcb..],
                B, T, C, 4 * C);
            GeluForward(
                model.ActsMemory.AsSpan()[acts_l_fch_gelu..],
                model.ActsMemory.AsSpan()[acts_l_fch..],
                B * T * 4 * C);
            MatmulForward(
                model.ActsMemory.AsSpan()[acts_l_fcproj..],
                model.ActsMemory.AsSpan()[acts_l_fch_gelu..],
                model.ParamMemory.AsSpan()[param_l_fcprojw..],
                model.ParamMemory.AsSpan()[param_l_fcprojb..],
                B, T, 4 * C, C);
            ResidualForward(
                model.ActsMemory.AsSpan()[acts_l_residual3..],
                model.ActsMemory.AsSpan()[acts_l_residual2..],
                model.ActsMemory.AsSpan()[acts_l_fcproj..],
                B * T * C);
        }
        acts_residual = acts.Residual3.Item1 + ((L - 1) * B * T * C); // last residual is in residual3
        LayernormForward(
            model.ActsMemory.AsSpan()[acts.Lnf.Item1..acts.Lnf.Item2],
            model.ActsMemory.AsSpan()[acts.LnfMean.Item1..acts.LnfMean.Item2],
            model.ActsMemory.AsSpan()[acts.LnfRstd.Item1..acts.LnfRstd.Item2],
            model.ActsMemory.AsSpan()[acts_residual..],
            model.ParamMemory.AsSpan()[param.Lnfw.Item1..param.Lnfw.Item2],
            model.ParamMemory.AsSpan()[param.Lnfb.Item1..param.Lnfb.Item2],
            B, T, C);
        MatmulForward(
            model.ActsMemory.AsSpan()[acts.Logits.Item1..acts.Logits.Item2],
            model.ActsMemory.AsSpan()[acts.Lnf.Item1..acts.Lnf.Item2],
            model.ParamMemory.AsSpan()[param.Wte.Item1..param.Wte.Item2],
            null, B, T, C, V);
        SoftmaxForward(
            model.ActsMemory.AsSpan()[acts.Probs.Item1..acts.Probs.Item2],
            model.ActsMemory.AsSpan()[acts.Logits.Item1..acts.Logits.Item2],
            B, T, V);

        // also forward the cross-entropy loss function if we have the targets
        if (targets != null)
        {
            CrossentropyForward(
                model.ActsMemory.AsSpan()[model.Acts.Losses.Item1..model.Acts.Losses.Item2],
                model.ActsMemory.AsSpan()[model.Acts.Probs.Item1..model.Acts.Probs.Item2],
                targets, B, T, V);
            // for convenience also evaluate the mean loss
            float mean_loss = 0.0f;
            for (int i = 0; i < B * T; i++) { mean_loss += model.ActsMemory[model.Acts.Losses.Item1 + i]; }
            mean_loss /= B * T;
            model.MeanLoss = mean_loss;
        }
        else
        {
            // if we don't have targets, we don't have a loss
            model.MeanLoss = -1.0f;
        }
        return true;
    }

    public static void GPT2ZeroGrad(GPT2 model)
    {
        if (model.GradsMemory != null)
        {
            Array.Clear(model.GradsMemory);
        }
        if (model.GradsActsMemory != null)
        {
            Array.Clear(model.GradsActsMemory);
        }
    }

    public static bool GPT2Backward(GPT2 model)
    {
        // double check we forwarded previously, with targets
        if (model.MeanLoss == -1.0f)
        {
            Console.WriteLine("Error: must forward with targets before backward\n");
            return false;
        }

        // lazily allocate the memory for gradients of the weights and activations, if needed
        if (model.GradsMemory == null)
        {
            model.GradsMemory = MallocAndPointParameters(model.ParamSizes);

            var p = 0;
            List<int> pos = [p];
            for (var i = 0; i < model.ParamSizes.Length; i++)
                pos.Add(p += model.ParamSizes[i]);

            model.Grads.Wte = (pos[0], pos[1]);
            model.Grads.Wpe = (pos[1], pos[2]);
            model.Grads.Ln1w = (pos[2], pos[3]); ;
            model.Grads.Ln1b = (pos[3], pos[4]);
            model.Grads.Qkvw = (pos[4], pos[5]);
            model.Grads.Qkvb = (pos[5], pos[6]);
            model.Grads.AttProjw = (pos[6], pos[7]);
            model.Grads.AttProjb = (pos[7], pos[8]);
            model.Grads.Ln2w = (pos[8], pos[9]);
            model.Grads.Ln2b = (pos[9], pos[10]);
            model.Grads.Fcw = (pos[10], pos[11]);
            model.Grads.Fcb = (pos[11], pos[12]);
            model.Grads.FcProjw = (pos[12], pos[13]);
            model.Grads.FcProjb = (pos[13], pos[14]);
            model.Grads.Lnfw = (pos[14], pos[15]);
            model.Grads.Lnfb = (pos[15], pos[16]);


            model.GradsActsMemory = MallocAndPointActivations(model.ActSizes);

            pos = [p = 0];
            for (var i = 0; i < model.ActSizes.Length; i++)
                pos.Add(p += model.ActSizes[i]);

            model.GradsActs.Encoded = (pos[0], pos[1]);
            model.GradsActs.Ln1 = (pos[1], pos[2]);
            model.GradsActs.Ln1Mean = (pos[2], pos[3]);
            model.GradsActs.Ln1Rstd = (pos[3], pos[4]);
            model.GradsActs.Qkv = (pos[4], pos[5]);
            model.GradsActs.Atty = (pos[5], pos[6]);
            model.GradsActs.Preatt = (pos[6], pos[7]);
            model.GradsActs.Att = (pos[7], pos[8]);
            model.GradsActs.AttProj = (pos[8], pos[9]);
            model.GradsActs.Residual2 = (pos[9], pos[10]);
            model.GradsActs.Ln2 = (pos[10], pos[11]);
            model.GradsActs.Ln2Mean = (pos[11], pos[12]);
            model.GradsActs.Ln2Rstd = (pos[12], pos[13]);
            model.GradsActs.Fch = (pos[13], pos[14]);
            model.GradsActs.FchGelu = (pos[14], pos[15]);
            model.GradsActs.FcProj = (pos[15], pos[16]);
            model.GradsActs.Residual3 = (pos[16], pos[17]);
            model.GradsActs.Lnf = (pos[17], pos[18]);
            model.GradsActs.LnfMean = (pos[18], pos[19]);
            model.GradsActs.LnfRstd = (pos[19], pos[20]);
            model.GradsActs.Logits = (pos[20], pos[21]);
            model.GradsActs.Probs = (pos[21], pos[22]);
            model.GradsActs.Losses = (pos[22], pos[23]);

            GPT2ZeroGrad(model);
        }

        // convenience shortcuts
        int B = model.BatchSize;
        int T = model.SeqLen;
        int V = model.Config.VocabSize;
        int L = model.Config.NumLayers;
        int NH = model.Config.NumHeads;
        int C = model.Config.Channels;

        // backward pass
        ParameterTensors param = model.Params; // for brevity
        ParameterTensors grads = model.Grads;
        ActivationTensors acts = model.Acts;
        ActivationTensors grads_acts = model.GradsActs;

        // we kick off the chain by filling in dlosses with 1.0f/(B*T), to get the mean loss
        float dloss_mean = 1.0f / (B * T);
        for (int i = 0; i < B * T; i++)
        {
            model.GradsActsMemory[grads_acts.Losses.Item1 + i]
                = dloss_mean;
        }

        CrossentropySoftmaxBackward(
            model.GradsActsMemory.AsSpan()[grads_acts.Logits.Item1..grads_acts.Logits.Item2],
            model.GradsActsMemory.AsSpan()[grads_acts.Losses.Item1..grads_acts.Losses.Item2],
            model.ActsMemory.AsSpan()[acts.Probs.Item1..acts.Probs.Item2],
            model.Targets, B, T, V);
        MatmulBackward(
            model.GradsActsMemory.AsSpan()[grads_acts.Lnf.Item1..grads_acts.Lnf.Item2],
            model.GradsMemory.AsSpan()[grads.Wte.Item1..grads.Wte.Item2],
            null,
            model.GradsActsMemory.AsSpan()[grads_acts.Logits.Item1..grads_acts.Logits.Item2],
            model.ActsMemory.AsSpan()[acts.Lnf.Item1..acts.Lnf.Item2],
            model.ParamMemory.AsSpan()[param.Wte.Item1..param.Wte.Item2],
            B, T, C, V);
        int acts_residual = acts.Residual3.Item1 + ((L - 1) * B * T * C); // last layer's residual
        int grads_acts_dresidual = grads_acts.Residual3.Item1 + ((L - 1) * B * T * C); // write to last layer's residual
        LayernormBackward(
            model.GradsActsMemory.AsSpan()[grads_acts_dresidual..],
            model.GradsMemory.AsSpan()[grads.Lnfw.Item1..grads.Lnfw.Item2],
            model.GradsMemory.AsSpan()[grads.Lnfb.Item1..grads.Lnfb.Item2],
            model.GradsActsMemory.AsSpan()[grads_acts.Lnf.Item1..grads_acts.Lnf.Item2],
            model.ActsMemory.AsSpan()[acts_residual..],
            model.ParamMemory.AsSpan()[param.Lnfw.Item1..param.Lnfw.Item2],
            model.ActsMemory.AsSpan()[acts.LnfMean.Item1..acts.LnfMean.Item2],
            model.ActsMemory.AsSpan()[acts.LnfRstd.Item1..acts.LnfRstd.Item2],
            B, T, C);

        for (int l = L - 1; l >= 0; l--)
        {

            acts_residual = l == 0 ? acts.Encoded.Item1 : acts.Residual3.Item1 + ((l - 1) * B * T * C);
            grads_acts_dresidual = l == 0 ? grads_acts.Encoded.Item1 : grads_acts.Residual3.Item1 + ((l - 1) * B * T * C);

            // get the pointers of the weights for this layer
            int param_l_ln1w = param.Ln1w.Item1 + (l * C);
            int param_l_qkvw = param.Qkvw.Item1 + (l * 3 * C * C);
            int param_l_attprojw = param.AttProjw.Item1 + (l * C * C);
            int param_l_ln2w = param.Ln2w.Item1 + (l * C);
            int param_l_fcw = param.Fcw.Item1 + (l * 4 * C * C);
            int param_l_fcprojw = param.FcProjw.Item1 + (l * C * 4 * C);
            // get the pointers of the gradients of the weights for this layer
            int grads_dl_ln1w = grads.Ln1w.Item1 + (l * C);
            int grads_dl_ln1b = grads.Ln1b.Item1 + (l * C);
            int grads_dl_qkvw = grads.Qkvw.Item1 + (l * 3 * C * C);
            int grads_dl_qkvb = grads.Qkvb.Item1 + (l * 3 * C);
            int grads_dl_attprojw = grads.AttProjw.Item1 + (l * C * C);
            int grads_dl_attprojb = grads.AttProjb.Item1 + (l * C);
            int grads_dl_ln2w = grads.Ln2w.Item1 + (l * C);
            int grads_dl_ln2b = grads.Ln2b.Item1 + (l * C);
            int grads_dl_fcw = grads.Fcw.Item1 + (l * 4 * C * C);
            int grads_dl_fcb = grads.Fcb.Item1 + (l * 4 * C);
            int grads_dl_fcprojw = grads.FcProjw.Item1 + (l * C * 4 * C);
            int grads_dl_fcprojb = grads.FcProjb.Item1 + (l * C);
            // get the pointers of the activations for this layer
            int acts_l_ln1 = acts.Ln1.Item1 + (l * B * T * C);
            int acts_l_ln1_mean = acts.Ln1Mean.Item1 + (l * B * T);
            int acts_l_ln1_rstd = acts.Ln1Rstd.Item1 + (l * B * T);
            int acts_l_qkv = acts.Qkv.Item1 + (l * B * T * 3 * C);
            int acts_l_atty = acts.Atty.Item1 + (l * B * T * C);
            int acts_l_att = acts.Att.Item1 + (l * B * NH * T * T);
            int acts_l_residual2 = acts.Residual2.Item1 + (l * B * T * C);
            int acts_l_ln2 = acts.Ln2.Item1 + (l * B * T * C);
            int acts_l_ln2_mean = acts.Ln2Mean.Item1 + (l * B * T);
            int acts_l_ln2_rstd = acts.Ln2Rstd.Item1 + (l * B * T);
            int acts_l_fch = acts.Fch.Item1 + (l * B * T * 4 * C);
            int acts_l_fch_gelu = acts.FchGelu.Item1 + (l * B * T * 4 * C);
            // get the pointers of the gradients of the activations for this layer
            int grads_acts_dl_ln1 = grads_acts.Ln1.Item1 + (l * B * T * C);
            int grads_acts_dl_qkv = grads_acts.Qkv.Item1 + (l * B * T * 3 * C);
            int grads_acts_dl_atty = grads_acts.Atty.Item1 + (l * B * T * C);
            int grads_acts_dl_preatt = grads_acts.Preatt.Item1 + (l * B * NH * T * T);
            int grads_acts_dl_att = grads_acts.Att.Item1 + (l * B * NH * T * T);
            int grads_acts_dl_attproj = grads_acts.AttProj.Item1 + (l * B * T * C);
            int grads_acts_dl_residual2 = grads_acts.Residual2.Item1 + (l * B * T * C);
            int grads_acts_dl_ln2 = grads_acts.Ln2.Item1 + (l * B * T * C);
            int grads_acts_dl_fch = grads_acts.Fch.Item1 + (l * B * T * 4 * C);
            int grads_acts_dl_fch_gelu = grads_acts.FchGelu.Item1 + (l * B * T * 4 * C);
            int grads_acts_dl_fcproj = grads_acts.FcProj.Item1 + (l * B * T * C);
            int grads_acts_dl_residual3 = grads_acts.Residual3.Item1 + (l * B * T * C);

            // backprop this layer
            ResidualBackward(
                model.GradsActsMemory.AsSpan()[grads_acts_dl_residual2..],
                model.GradsActsMemory.AsSpan()[grads_acts_dl_fcproj..],
                model.GradsActsMemory.AsSpan()[grads_acts_dl_residual3..],
                B * T * C);
            MatmulBackward(
                model.GradsActsMemory.AsSpan()[grads_acts_dl_fch_gelu..],
                model.GradsMemory.AsSpan()[grads_dl_fcprojw..],
                model.GradsMemory.AsSpan()[grads_dl_fcprojb..],
                model.GradsActsMemory.AsSpan()[grads_acts_dl_fcproj..],
                model.ActsMemory.AsSpan()[acts_l_fch_gelu..],
                model.ParamMemory.AsSpan()[param_l_fcprojw..],
                B, T, 4 * C, C);
            GeluBackward(
                model.GradsActsMemory.AsSpan()[grads_acts_dl_fch..],
                model.ActsMemory.AsSpan()[acts_l_fch..],
                model.GradsActsMemory.AsSpan()[grads_acts_dl_fch_gelu..],
                B * T * 4 * C);
            MatmulBackward(
                model.GradsActsMemory.AsSpan()[grads_acts_dl_ln2..],
                model.GradsMemory.AsSpan()[grads_dl_fcw..],
                model.GradsMemory.AsSpan()[grads_dl_fcb..],
                model.GradsActsMemory.AsSpan()[grads_acts_dl_fch..],
                model.ActsMemory.AsSpan()[acts_l_ln2..],
                model.ParamMemory.AsSpan()[param_l_fcw..],
                B, T, C, 4 * C);
            LayernormBackward(
                model.GradsActsMemory.AsSpan()[grads_acts_dl_residual2..],
                model.GradsMemory.AsSpan()[grads_dl_ln2w..],
                model.GradsMemory.AsSpan()[grads_dl_ln2b..],
                model.GradsActsMemory.AsSpan()[grads_acts_dl_ln2..],
                model.ActsMemory.AsSpan()[acts_l_residual2..],
                model.ParamMemory.AsSpan()[param_l_ln2w..],
                model.ActsMemory.AsSpan()[acts_l_ln2_mean..],
                model.ActsMemory.AsSpan()[acts_l_ln2_rstd..],
                B, T, C);
            ResidualBackward(
                model.GradsActsMemory.AsSpan()[grads_acts_dresidual..],
                model.GradsActsMemory.AsSpan()[grads_acts_dl_attproj..],
                model.GradsActsMemory.AsSpan()[grads_acts_dl_residual2..],
                B * T * C);
            MatmulBackward(
                model.GradsActsMemory.AsSpan()[grads_acts_dl_atty..],
                model.GradsMemory.AsSpan()[grads_dl_attprojw..],
                model.GradsMemory.AsSpan()[grads_dl_attprojb..],
                model.GradsActsMemory.AsSpan()[grads_acts_dl_attproj..],
                model.ActsMemory.AsSpan()[acts_l_atty..],
                model.ParamMemory.AsSpan()[param_l_attprojw..],
                B, T, C, C);
            AttentionBackward(
                model.GradsActsMemory.AsSpan()[grads_acts_dl_qkv..],
                model.GradsActsMemory.AsSpan()[grads_acts_dl_preatt..],
                model.GradsActsMemory.AsSpan()[grads_acts_dl_att..],
                model.GradsActsMemory.AsSpan()[grads_acts_dl_atty..],
                model.ActsMemory.AsSpan()[acts_l_qkv..],
                model.ActsMemory.AsSpan()[acts_l_att..],
                B, T, C, NH);
            MatmulBackward(
                model.GradsActsMemory.AsSpan()[grads_acts_dl_ln1..],
                model.GradsMemory.AsSpan()[grads_dl_qkvw..],
                model.GradsMemory.AsSpan()[grads_dl_qkvb..],
                model.GradsActsMemory.AsSpan()[grads_acts_dl_qkv..],
                model.ActsMemory.AsSpan()[acts_l_ln1..],
                model.ParamMemory.AsSpan()[param_l_qkvw..],
                B, T, C, 3 * C);
            LayernormBackward(
                model.GradsActsMemory.AsSpan()[grads_acts_dresidual..],
                model.GradsMemory.AsSpan()[grads_dl_ln1w..],
                model.GradsMemory.AsSpan()[grads_dl_ln1b..],
                model.GradsActsMemory.AsSpan()[grads_acts_dl_ln1..],
                model.ActsMemory.AsSpan()[acts_residual..],
                model.ParamMemory.AsSpan()[param_l_ln1w..],
                model.ActsMemory.AsSpan()[acts_l_ln1_mean..],
                model.ActsMemory.AsSpan()[acts_l_ln1_rstd..],
                B, T, C);
        }
        EncoderBackward(
            model.GradsMemory.AsSpan()[grads.Wte.Item1..grads.Wte.Item2],
            model.GradsMemory.AsSpan()[grads.Wpe.Item1..grads.Wpe.Item2],
            model.GradsActsMemory.AsSpan()[grads_acts.Encoded.Item1..grads_acts.Encoded.Item2],
            model.Inputs, B, T, C);
        return true;
    }

    public static void GPT2Update(GPT2 model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t)
    {
        // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

        // lazily allocate the memory for m_memory and v_memory
        if (model.M_Memory == null)
        {
            model.M_Memory = (new float[model.NumParameters]);
            model.V_Memory = (new float[model.NumParameters]);
        }

        for (int i = 0; i < model.NumParameters; i++)
        {
            float param = model.ParamMemory[i];
            float grads = model.GradsMemory[i];

            // update the first moment (momentum)
            float m = beta1 * model.M_Memory[i] + (1.0f - beta1) * grads;
            // update the second moment (RMSprop)
            float v = beta2 * model.V_Memory[i] + (1.0f - beta2) * grads * grads;
            // bias-correct both moments
            float m_hat = m / (1.0f - (float)Math.Pow(beta1, t));
            float v_hat = v / (1.0f - (float)Math.Pow(beta2, t));

            // update
            model.M_Memory[i] = m;
            model.V_Memory[i] = v;
            model.ParamMemory[i] -= learning_rate * (m_hat / ((float)Math.Sqrt(v_hat) + eps) + weight_decay * param);
        }
    }

    // ----------------------------------------------------------------------------
    // samples
    public const int GPT2_EOT = 50256;

    public static ulong RandomU32(ref ulong state)
    {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return (state * 0x2545F4914F6CDD1Dul) >> 32;
    }
    public static float RandomF32(ref ulong state) 
        => (RandomU32(ref state) >> 8) / 16777216.0f;

    public static int SampleMult(Span<float> probabilities, int n, float coin)
    {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from random_f32()
        float cdf = 0.0f;
        for (int i = 0; i < n; i++)
        {
            cdf += probabilities[i];
            if (coin < cdf)
            {
                return i;
            }
        }
        return n - 1; // in case of rounding errors
    }

    public const int GEN_MAX_LENGTH = 64;
    // ----------------------------------------------------------------------------
    // main training loop
    public static bool Train(string training_file)
    {
        Console.WriteLine($"Training starting at {DateTime.Now}...");
        var global_watch = new Stopwatch();
        global_watch.Start();

        // build the GPT-2 model from a checkpoint
        var model = GPT2BuildFromCheckpoint(training_file);

        // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
        var tiny_stories_train = "data/TinyStories_train.bin";
        var tiny_stories_val = "data/TinyStories_val.bin";
        var tiny_shakespeare_train = "data/tiny_shakespeare_train.bin";
        var tiny_shakespeare_val = "data/tiny_shakespeare_val.bin";
        var train_tokens = File.Exists(tiny_shakespeare_train) ? tiny_shakespeare_train : tiny_stories_train;
        var val_tokens = File.Exists(tiny_shakespeare_val) ? tiny_shakespeare_val : tiny_stories_val;
        var B = 4;
        var T = 64;
        using var train_loader = new DataLoader();
        train_loader.Init(train_tokens, B, T);
        Console.WriteLine("train dataset num_batches: {0}", train_loader.NumBatches);
        using var val_loader = new DataLoader();
        val_loader.Init(val_tokens, B, T);
        Console.WriteLine("val dataset num_batches: {0}", val_loader.NumBatches);
        int val_num_batches = 10;

        // some memory for generating samples from the model
        ulong rng_state = 1337;
        var gen_tokens = new int[GEN_MAX_LENGTH];
        const int max_step = 40;
        // train
        //timespec start, end;
        for (var step = 0; step <= max_step; step++)
        {
            Console.WriteLine($"training progress {step}/{max_step}...");
            // once in a while estimate the validation loss
            if (step % 10 == 0)
            {
                float val_loss = 0.0f;
                val_loader.Reset();
                for (int i = 0; i < val_num_batches; i++)
                {
                    val_loader.NextBatch();
                    GPT2Forward(model, val_loader.Inputs, val_loader.Targets, B, T);
                    val_loss += model.MeanLoss;
                    Console.WriteLine("\tbatch {0} mean loss {1}", i, model.MeanLoss);
                }
                val_loss /= val_num_batches;
                Console.WriteLine("val loss {0}", val_loss);
            }

            // once in a while do model inference to print generated text
            if (step > 0 && step % 20 == 0)
            {
                gen_tokens[0] = GPT2_EOT; // the GPT-2 EOT token kicks off the generation
                for (int t = 1; t < GEN_MAX_LENGTH; t++)
                {
                    // note that inference is wasteful here because
                    // for each t, we re-compute all activations between 0 and t
                    // leaving this alone because you want separate code for inference anyway
                    // the inference here is just for sanity checking purposes
                    GPT2Forward(model, gen_tokens, null, 1, t);
                    Span<float> probs = model.ActsMemory.AsSpan()[(model.Acts.Probs.Item1 + ((t - 1) * model.Config.VocabSize))..];
                    float coin = RandomF32(ref rng_state);
                    int next_token = SampleMult(probs, model.Config.VocabSize, coin);
                    gen_tokens[t] = next_token;
                    Console.WriteLine($"generating token {t}/{GEN_MAX_LENGTH}...");
                }
                Console.WriteLine("generated: ");
                for (int t = 0; t < GEN_MAX_LENGTH; t++)
                {
                    Console.WriteLine("{0} ", gen_tokens[t]);
                }
                Console.WriteLine();
            }

            // do a training step

            var watch = new Stopwatch();
            watch.Start();
            train_loader.NextBatch();
            GPT2Forward(model, train_loader.Inputs, train_loader.Targets, B, T);
            GPT2ZeroGrad(model);
            GPT2Backward(model);
            GPT2Update(model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);
            watch.Stop();
            Console.WriteLine("step {0}: train loss {1} (took {2} ms)", step, model.MeanLoss, watch.Elapsed.Milliseconds);
        }
        global_watch.Stop();
        Console.WriteLine($"Total {global_watch.Elapsed} seconds used for training.");

        return true;
    }
}
