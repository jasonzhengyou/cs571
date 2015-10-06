package edu.emory.mathcs.nlp.deeplearning.network;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.List;
import java.util.function.Consumer;

import org.kohsuke.args4j.Option;

import edu.emory.mathcs.nlp.common.util.BinUtils;
import edu.emory.mathcs.nlp.common.util.FileUtils;
import edu.emory.mathcs.nlp.common.util.IOUtils;
import edu.emory.mathcs.nlp.component.util.NLPComponent;
import edu.emory.mathcs.nlp.component.util.NLPFlag;
import edu.emory.mathcs.nlp.component.util.config.NLPConfig;
import edu.emory.mathcs.nlp.component.util.eval.Eval;
import edu.emory.mathcs.nlp.component.util.feature.FeatureTemplate;
import edu.emory.mathcs.nlp.component.util.reader.TSVReader;
import edu.emory.mathcs.nlp.component.util.state.NLPState;
import edu.emory.mathcs.nlp.component.util.train.Aggregation;
import edu.emory.mathcs.nlp.learn.model.StringModel;
import edu.emory.mathcs.nlp.learn.optimization.Optimizer;
import edu.emory.mathcs.nlp.learn.optimization.OptimizerType;

public abstract class FeedForwardNeuralNetworkTrain<N,S extends NLPState<N>> {
	@Option(name="-c", usage="confinguration file (required)", required=true, metaVar="<filename>")
	public String configuration_file;
	@Option(name="-t", usage="training path (required)", required=true, metaVar="<filepath>")
	public String train_path;
	@Option(name="-te", usage="training file extension (default: *)", required=false, metaVar="<string>")
	public String train_ext = "*";
	@Option(name="-d", usage="development path (required)", required=true, metaVar="<filepath>")
	public String develop_path;
	@Option(name="-de", usage="development file extension (default: *)", required=false, metaVar="<string>")
	public String develop_ext = "*";
	@Option(name="-f", usage="feature template ID (default: 0)", required=false, metaVar="integer")
	public int feature_template = 0;
	@Option(name="-m", usage="model file (optional)", required=false, metaVar="<filename>")
	public String model_file = null;
	
	public FeedForwardNeuralNetworkTrain() {};
	
	public FeedForwardNeuralNetworkTrain(String[] args)
	{
		BinUtils.initArgs(args, this);
	}
	
	/** Collects necessary lexicons for the component before training. */
	public abstract void collect(TSVReader<N> reader, List<String> inputFiles, NLPComponent<N,S> component, NLPConfig<N> configuration);
	protected abstract NLPConfig<N> createConfiguration(String filename);
	protected abstract FeatureTemplate<N,S> createFeatureTemplate();
	protected abstract NLPComponent<N,S> createComponent();
	protected abstract Eval createEvaluator();
	
    public void train()
    {
        List<String>        trainFiles    = FileUtils.getFileList(train_path, train_ext);
        List<String>        developFiles  = FileUtils.getFileList(develop_path, develop_ext);
        NLPConfig<N>        configuration = createConfiguration(configuration_file);
        TSVReader<N>        reader        = configuration.getTSVReader();
        NLPComponent<N,S> component     = createComponent();

        component.setFeatureTemplate(createFeatureTemplate());
        component.setEval(createEvaluator());

        train(reader, trainFiles, developFiles, configuration, component);
        if (model_file != null) save(component);
    }
    
	public void train(TSVReader<N> reader, List<String> trainFiles, List<String> developFiles, NLPConfig<N> configuration, NLPComponent<N,S> component)
	{
		BinUtils.LOG.info("Collecting lexicons:\n");
		collect(reader, trainFiles, component, configuration);
		
		Aggregation dagger = configuration.getAggregation();
		StringModel[] models = component.getModels();
		int i, size = models.length, bestIter = 0;
		float[][] bestWeight = new float[size][];
		double prevScore, currScore = -1, bestScore = -1;
		
		for (int iter=0; ; iter++)
		{
			BinUtils.LOG.info(String.format("\nTraining: %d\n\n", iter));
			component.setFlag(iter == 0 ? NLPFlag.TRAIN : NLPFlag.AGGREGATE);
			iterate(reader, trainFiles, component::process);
			
			component.setFlag(NLPFlag.EVALUATE);
			prevScore = currScore;
			currScore = train(reader, developFiles, component, configuration);
			if (dagger == null) break;	// no aggregating
			
			if (prevScore >= currScore + dagger.getToleranceDelta() || iter - dagger.getMaxTolerance() > bestIter)
			{
				for (i=0; i<size; i++) models[i].getWeightVector().fromArray(bestWeight[i]);
				break;
			}
			else if (bestScore < currScore)
			{
				for (i=0; i<size; i++) bestWeight[i] = models[i].getWeightVector().toArray().clone();
				bestScore = currScore;
				bestIter  = iter;
			}
		}
		
		BinUtils.LOG.info(String.format("\nFinal score: %5.2f\n", bestScore));
	}
	//changed from NLPTrain********************** 
	public double train(TSVReader<N> reader, List<String> developFiles, NLPComponent<N,?> component, NLPConfig<N> configuration)
	{
		StringModel[] models = component.getModels();
		FeedForwardNeuralNetwork Ffnn = configuration.getNeuralNetwork(models[0], 0);
		models[0].setNeuralNetwork(Ffnn);
		BinUtils.LOG.info(Ffnn.toString() + " \n");
		BinUtils.LOG.info(models[0].trainInfo()+"\n");
		
		//only doing trainOnline since that is how the flag is set
		double score = trainOnline(reader, developFiles, component, Ffnn, models[0]);
		return score;
	}


    /** Called by {@link #train(TSVReader, List, NLPComponent, NLPConfig)}. */
    protected double trainOnline(TSVReader<N> reader, List<String> developFiles, NLPComponent<N,?> component,
                                   FeedForwardNeuralNetwork optimizer, StringModel model)
    {
        Eval eval = component.getEval();
        double currScore, bestScore = 0, prevScore = 0;
		float[] prevWeight = model.getWeightVector().toArray();

		for (int epoch=1; ;epoch++)
		{
			eval.clear();
			optimizer.train(model.getInstanceList());
			iterate(reader, developFiles, nodes -> component.process(nodes));
			currScore = eval.score();
			
			if (prevScore < currScore)
			{
				prevScore  = currScore;
				prevWeight = model.getWeightVector().toArray().clone();
			}
			else
			{
				model.getWeightVector().fromArray(prevWeight);
				break;
			}
			
			BinUtils.LOG.info(String.format("%3d: %5.2f\n", epoch, currScore));
		}
		
		return prevScore;
    }

//	=================================== HELPERS ===================================

    protected void iterate(TSVReader<N> reader, List<String> inputFiles, Consumer<N[]> f)
    {
        N[] nodes;

        for (String inputFile : inputFiles)
        {
            reader.open(IOUtils.createFileInputStream(inputFile));

            try
            {
                while ((nodes = reader.next()) != null)
                    f.accept(nodes);
            }
            catch (IOException e) {e.printStackTrace();}
        }
    }

    public void save(NLPComponent<N,S> component)
    {
        ObjectOutputStream out = IOUtils.createObjectXZBufferedOutputStream(model_file);

        try
        {
            out.writeObject(component);
            out.close();
        }
        catch (IOException e) {e.printStackTrace();}
    }
	
}
