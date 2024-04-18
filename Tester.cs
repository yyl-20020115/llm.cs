using System.Diagnostics;

namespace llm.cs;

public class Tester : Trainer
{
    public static bool CheckTensor(Span<float> a, Span<float> b, int n, string label)
    {
        if (a.Length != n || b.Length != n) return false;
        var print_upto = 5;
        var ok = true;
        Console.WriteLine(label);
        for (int i = 0; i < n; i++)
        {
            if (Math.Abs(a[i] - b[i]) <= 1e-2)
            {
                if (i < print_upto) { Console.WriteLine("OK "); }
            }
            else
            {
                if (i < print_upto) { Console.WriteLine("NOT OK "); }
                ok = false;
            }
            if (i < print_upto) { Console.WriteLine("{0} {1}", a[i], b[i]); }
        }
        // print the final result
        if (ok)
        {
            Console.WriteLine("TENSOR OK");
        }
        else
        {
            Console.WriteLine("TENSOR NOT OK");
        }
        return ok;
    }
    public const int max_steps = 10;

    public static bool Test(string test_file, string state_file)
    {
        // build the GPT-2 model from a checkpoint
        var model = GPT2BuildFromCheckpoint(test_file);

        var C = model.Config.Channels;
        var V = model.Config.VocabSize;
        var maxT = model.Config.MaxSeqLen;
        var L = model.Config.NumLayers;

        // load additional information that we will use for debugging and error checking
        if (!File.Exists(state_file))
        {
            Console.WriteLine("Error opening state file");
            return false;
        }

        using var original_state_stream = new FileStream(state_file, FileMode.Open);
        using var original_state_reader = new BinaryReader(original_state_stream);

        var state_header = new int[256];
        for (int i = 0; i < state_header.Length; i++)
        {
            state_header[i] = original_state_reader.ReadInt32();
        }

        if (state_header[0] != 20240327)
        {
            Console.WriteLine("Bad magic state file");
            return false;
        }
        if (state_header[1] != 1)
        {
            Console.WriteLine("Bad version in state file");
            return false;
        }
        var B = state_header[2]; // batch size, e.g. 4
        var T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
        Console.WriteLine("[State]");
        Console.WriteLine("batch_size: {0}", B);
        Console.WriteLine("seq_len: {0}", T);

        var expected_grads = new ParameterTensors();
        var expected_grads_memory = MallocAndPointParameters(model.ParamSizes);
        //expected_grads
        // inputs and expected outputs, only used for error checking
        var input = new int[(B * T)];
        var output = new int[(B * T)];
        var expected_logits = new float[(B * T * V)];
        float expected_loss = 0;
        float actual_loss = 0;

        var actual_logits = new float[expected_logits.Length];
        var actual_losses = new float[max_steps];
        var actual_grads_memory = new float[expected_grads_memory.Length];

        for (var i = 0; i < input.Length; i++)
        {
            input[i] = original_state_reader.ReadInt32();
        }
        for (var i = 0; i < output.Length; i++)
        {
            output[i] = original_state_reader.ReadInt32();
        }
        for (var i = 0; i < expected_logits.Length; i++)
        {
            expected_logits[i] = original_state_reader.ReadSingle();
        }
        expected_loss = original_state_reader.ReadSingle();
        // read reference information from Python
        for (int i = 0; i < model.NumParameters; i++)
        {
            expected_grads_memory[i] = original_state_reader.ReadSingle();
        }
        var p = 0;
        var pos = new List<int> { 0 };
        for (var i = 0; i < model.ParamSizes.Length; i++)
            pos.Add(p += model.ParamSizes[i]);
        expected_grads.Wte = (pos[0], pos[1]);
        expected_grads.Wpe = (pos[1], pos[2]);
        expected_grads.Ln1w = (pos[2], pos[3]);
        expected_grads.Ln1b = (pos[3], pos[4]);
        expected_grads.Qkvw = (pos[4], pos[5]);
        expected_grads.Qkvb = (pos[5], pos[6]);
        expected_grads.AttProjw = (pos[6], pos[7]);
        expected_grads.AttProjb = (pos[7], pos[8]);
        expected_grads.Ln2w = (pos[8], pos[9]);
        expected_grads.Ln2b = (pos[9], pos[10]);
        expected_grads.Fcw = (pos[10], pos[11]);
        expected_grads.Fcb = (pos[11], pos[12]);
        expected_grads.FcProjw = (pos[12], pos[13]);
        expected_grads.FcProjb = (pos[13], pos[14]);
        expected_grads.Lnfw = (pos[14], pos[15]);
        expected_grads.Lnfb = (pos[15], pos[16]);

        // overall OK signal for the test
        var allok = true;
        // let's do 10 training iterations, following the pytorch code
        var losses = new float[max_steps];
        for (int step = 0; step < losses.Length; step++)
        {
            var watch = new Stopwatch();
            watch.Start();
            GPT2Forward(model, input, output, B, T);
            GPT2ZeroGrad(model);
            GPT2Backward(model);
            watch.Stop();

            if (step == 0)
            {
                // error checking at step 0 for reference activations/gradients

                // at this point, target should be equal to expected_logits, let's compare
                var logits_ok = true;
                for (var i = 0; i < B * T * V; i++)
                {
                    actual_logits[i] = model.ActsMemory[model.Acts.Logits.Item1 + i];

                    if (i < 3)
                    {
                        Console.WriteLine("{0} {1}", expected_logits[i], model.ActsMemory[model.Acts.Logits.Item1 + i]);
                    }
                    if (Math.Abs(expected_logits[i] - model.ActsMemory[model.Acts.Logits.Item1 + i]) >= 1e-2)
                    {
                        Console.WriteLine("MISMATCH AT INDEX {0}: ", i);
                        Console.WriteLine("{0} {1}", expected_logits[i], model.ActsMemory[model.Acts.Logits.Item1 + i]);
                        logits_ok = false;
                        break;
                    }
                }
                if (!logits_ok) { Console.WriteLine("NOT "); }
                Console.WriteLine("OK (LOGITS)");
                allok = allok && logits_ok;

                actual_loss = model.MeanLoss;
                // compare the achieved loss
                if (Math.Abs(actual_loss - expected_loss) >= 1e-2)
                {
                    Console.WriteLine("LOSS MISMATCH: {0} {1}", actual_loss, expected_loss);
                    allok = false;
                }
                else
                {
                    Console.WriteLine("LOSS OK: {0} {1}", actual_loss, expected_loss);
                }

                // finally check all the gradients
                var gradoks = new bool[16];
                var grads = model.Grads;
                gradoks[0] = CheckTensor(model.GradsMemory.AsSpan()[grads.Wte.Item1..grads.Wte.Item2], expected_grads_memory.AsSpan()[expected_grads.Wte.Item1..expected_grads.Wte.Item2], V * C, "dwte");
                gradoks[1] = CheckTensor(model.GradsMemory.AsSpan()[grads.Wpe.Item1..grads.Wpe.Item2], expected_grads_memory.AsSpan()[expected_grads.Wpe.Item1..expected_grads.Wpe.Item2], maxT * C, "dwpe");
                gradoks[2] = CheckTensor(model.GradsMemory.AsSpan()[grads.Ln1w.Item1..grads.Ln1w.Item2], expected_grads_memory.AsSpan()[expected_grads.Ln1w.Item1..expected_grads.Ln1w.Item2], L * C, "dln1w");
                gradoks[3] = CheckTensor(model.GradsMemory.AsSpan()[grads.Ln1b.Item1..grads.Ln1b.Item2], expected_grads_memory.AsSpan()[expected_grads.Ln1b.Item1..expected_grads.Ln1b.Item2], L * C, "dln1b");
                gradoks[4] = CheckTensor(model.GradsMemory.AsSpan()[grads.Qkvw.Item1..grads.Qkvw.Item2], expected_grads_memory.AsSpan()[expected_grads.Qkvw.Item1..expected_grads.Qkvw.Item2], L * 3 * C * C, "dqkvw");
                gradoks[5] = CheckTensor(model.GradsMemory.AsSpan()[grads.Qkvb.Item1..grads.Qkvb.Item2], expected_grads_memory.AsSpan()[expected_grads.Qkvb.Item1..expected_grads.Qkvb.Item2], L * 3 * C, "dqkvb");
                gradoks[6] = CheckTensor(model.GradsMemory.AsSpan()[grads.AttProjw.Item1..grads.AttProjw.Item2], expected_grads_memory.AsSpan()[expected_grads.AttProjw.Item1..expected_grads.AttProjw.Item2], L * C * C, "dattprojw");
                gradoks[7] = CheckTensor(model.GradsMemory.AsSpan()[grads.AttProjb.Item1..grads.AttProjb.Item2], expected_grads_memory.AsSpan()[expected_grads.AttProjb.Item1..expected_grads.AttProjb.Item2], L * C, "dattprojb");
                gradoks[8] = CheckTensor(model.GradsMemory.AsSpan()[grads.Ln2w.Item1..grads.Ln2w.Item2], expected_grads_memory.AsSpan()[expected_grads.Ln2w.Item1..expected_grads.Ln2w.Item2], L * C, "dln2w");
                gradoks[9] = CheckTensor(model.GradsMemory.AsSpan()[grads.Ln2b.Item1..grads.Ln2b.Item2], expected_grads_memory.AsSpan()[expected_grads.Ln2b.Item1..expected_grads.Ln2b.Item2], L * C, "dln2b");
                gradoks[10] = CheckTensor(model.GradsMemory.AsSpan()[grads.Fcw.Item1..grads.Fcw.Item2], expected_grads_memory.AsSpan()[expected_grads.Fcw.Item1..expected_grads.Fcw.Item2], L * 4 * C * C, "dfcw");
                gradoks[11] = CheckTensor(model.GradsMemory.AsSpan()[grads.Fcb.Item1..grads.Fcb.Item2], expected_grads_memory.AsSpan()[expected_grads.Fcb.Item1..expected_grads.Fcb.Item2], L * 4 * C, "dfcb");
                gradoks[12] = CheckTensor(model.GradsMemory.AsSpan()[grads.FcProjw.Item1..grads.FcProjw.Item2], expected_grads_memory.AsSpan()[expected_grads.FcProjw.Item1..expected_grads.FcProjw.Item2], L * C * 4 * C, "dfcprojw");
                gradoks[13] = CheckTensor(model.GradsMemory.AsSpan()[grads.FcProjb.Item1..grads.FcProjb.Item2], expected_grads_memory.AsSpan()[expected_grads.FcProjb.Item1..expected_grads.FcProjb.Item2], L * C, "dfcprojb");
                gradoks[14] = CheckTensor(model.GradsMemory.AsSpan()[grads.Lnfw.Item1..grads.Lnfw.Item2], expected_grads_memory.AsSpan()[expected_grads.Lnfw.Item1..expected_grads.Lnfw.Item2], C, "dlnfw");
                gradoks[15] = CheckTensor(model.GradsMemory.AsSpan()[grads.Lnfb.Item1..grads.Lnfb.Item2], expected_grads_memory.AsSpan()[expected_grads.Lnfb.Item1..expected_grads.Lnfb.Item2], C, "dlnfb");
                for (int i = 0; i < 16; i++)
                {
                    allok = allok && gradoks[i];
                }
                for (int i = 0; i < model.GradsMemory.Length; i++)
                {
                    actual_grads_memory[i] = model.GradsMemory[i];
                }
            }

            GPT2Update(model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);

            // print the timing information at the end
            Console.WriteLine("step {0}: loss {1} (took {2} ms)", step, model.MeanLoss, watch.Elapsed.Milliseconds);
            actual_losses[step] = losses[step] = model.MeanLoss;
        }

        // expected losses are as follows, from Python
#if FROM_PYTHON
        float[] expected_losses = [
                5.270007133483887f,
                4.059706687927246f,
                3.3751230239868164f,
                2.8007826805114746f,
                2.315382242202759f,
                1.8490285873413086f,
                1.3946564197540283f,
                0.9991465210914612f,
                0.6240804195404053f,
                0.37651097774505615f
            ];
#else
        float[] expected_losses = [
            5.2698894f,
            4.166697f,
            3.4952404f,
            2.9815965f,
            2.54753f,
            2.127743f,
            1.6642165f,
            1.2446052f,
            0.85817003f,
            0.5951968f
            ];
#endif
        for (int i = 0; i < max_steps; i++)
        {
            if (Math.Abs(losses[i] - expected_losses[i]) >= 2e-1)
            {
                Console.WriteLine("LOSS MISMATCH AT STEP {0}: {1} {2}", i, losses[i], expected_losses[i]);
                allok = false;
            }
            else
            {
                Console.WriteLine("LOSS OK AT STEP {0}: {1} {2}", i, losses[i], expected_losses[i]);
            }
        }

        Console.WriteLine("Overall Okay: {0}", allok);
        //write out the generated 
        var generate_file = Path.GetFileNameWithoutExtension(test_file) + ".gen.bin";
        using var generate_state_stream = new FileStream(generate_file, FileMode.Create);
        using var generate_state_writer = new BinaryWriter(generate_state_stream);
        for (int i = 0; i < state_header.Length; i++)
        {
            generate_state_writer.Write(state_header[i]);
        }
        for (int i = 0; i < input.Length; i++)
        {
            generate_state_writer.Write(input[i]);
        }
        for (int i = 0; i < output.Length; i++)
        {
            generate_state_writer.Write(output[i]);
        }
        for (var i = 0; i < actual_logits.Length; i++)
        {
            generate_state_writer.Write(actual_logits[i]);
        }
        generate_state_writer.Write(actual_loss);
        // read reference information from Python
        for (int i = 0; i < model.NumParameters; i++)
        {
            generate_state_writer.Write(actual_grads_memory[i]);
        }


        return true;
    }

}
