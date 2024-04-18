namespace llm.cs;

// if we are TESTING (see test_gpt2.c), we'll skip the int main below

// ----------------------------------------------------------------------------
// data loader lite
// returns random batches of data from a file of integers

public class DataLoader :IDisposable
{
    // hyperparameters
    public int B;
    public int T;
    // input handling and its state
    public FileStream? TokensFile;
    public long FileSize;
    public long CurrentPosition;
    // output memory
    public int[] Batch;
    public int[] Inputs;
    public int[] Targets;
    // convenience variables
    public long NumBatches;
    public bool Init(string filename, int B, int T)
    {
        this.B = B;
        this.T = T;
        if (!File.Exists(filename))
        {
            Console.WriteLine("Error opening tokens file\n");
            return false;
        }
        // open the input file for reading
        this.TokensFile = File.OpenRead(filename);
        // determine the file size
        this.TokensFile.Seek(0, SeekOrigin.End);
        this.FileSize = this.TokensFile.Position;
        this.TokensFile.Seek(0, SeekOrigin.Begin);
        if (this.FileSize < (B * T + 1) * sizeof(int))
        {
            Console.WriteLine("Error: file size is too small for the batch size and sequence length");
            return false;
        }
        this.CurrentPosition = 0; // start at the beginning

        // allocate space for B*T + 1 integers to store the inputs and targets
        this.Batch = (new int[(B * T + 1)]);
        this.NumBatches = this.FileSize / (B * T * sizeof(int));
        return true;
    }

    public void Reset() => CurrentPosition = 0;

    public void NextBatch()
    {
        if (this.TokensFile != null)
        {
            int B = this.B;
            int T = this.T;
            // if we are at the end of the file, loop back to the beginning
            if (this.CurrentPosition + (B * T + 1) * sizeof(int) > this.FileSize)
            {
                this.CurrentPosition = 0;
            }
            // read the B*T+1 integers from the file into batch
            this.TokensFile.Seek(this.CurrentPosition, SeekOrigin.Begin);
            //fseek(loader.tokens_file, loader.current_position, SEEK_SET);
            using var reader = new BinaryReader(this.TokensFile, System.Text.Encoding.Default, true);
            for (int i = 0; i < B * T + 1; i++)   
            {
                this.Batch[i] = reader.ReadInt32();
            }
            //fread(loader.batch, sizeof(int), B * T + 1, loader.tokens_file);
            // advance the current position by B*T integers
            this.CurrentPosition += B * T * sizeof(int);
            this.Inputs = this.Batch;
            this.Targets = this.Inputs[1..]; // targets are shifted by one
        }
    }

    public void Dispose()
    {
        this.TokensFile?.Close();
        this.TokensFile = null;
        this.Batch = [];
    }
}

