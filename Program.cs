namespace llm.cs;

public class Program
{
    public static bool TestOnly = true;
    public static void Main(string[] args)
    {
        var tf = args.Length > 0 ? args[0] : "gpt2_124M.bin";
        var sf = args.Length > 1 ? args[1] : "gpt2_124M_debug_state.bin";

        if (!TestOnly)
        {
            Trainer.Train(tf);
        }
        Tester.Test(tf,sf);
        Console.ReadLine();
    }
}
