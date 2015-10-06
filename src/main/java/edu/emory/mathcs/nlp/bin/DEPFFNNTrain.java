package edu.emory.mathcs.nlp.bin;

import java.util.List;

import edu.emory.mathcs.nlp.common.util.IOUtils;
import edu.emory.mathcs.nlp.component.dep.DEPConfig;
import edu.emory.mathcs.nlp.component.dep.DEPEval;
import edu.emory.mathcs.nlp.component.dep.DEPNode;
import edu.emory.mathcs.nlp.component.dep.DEPParser;
import edu.emory.mathcs.nlp.component.dep.DEPState;
import edu.emory.mathcs.nlp.component.dep.feature.DEPFeatureTemplate0;
import edu.emory.mathcs.nlp.component.dep.feature.DEPFeatureTemplate1;
import edu.emory.mathcs.nlp.component.dep.feature.DEPFeatureTemplate2;
import edu.emory.mathcs.nlp.component.util.NLPComponent;
import edu.emory.mathcs.nlp.component.util.config.NLPConfig;
import edu.emory.mathcs.nlp.component.util.eval.Eval;
import edu.emory.mathcs.nlp.component.util.feature.FeatureTemplate;
import edu.emory.mathcs.nlp.component.util.reader.TSVReader;
import edu.emory.mathcs.nlp.component.util.state.NLPState;
import edu.emory.mathcs.nlp.deeplearning.network.FeedForwardNeuralNetworkTrain;
import edu.emory.mathcs.nlp.learn.model.StringModel;
import edu.emory.mathcs.nlp.learn.weight.MultinomialWeightVector;

//Dependency Parsing Feed Forward Neural Network Training
public class DEPFFNNTrain extends FeedForwardNeuralNetworkTrain<DEPNode, DEPState<DEPNode>>{ //DEPNode,String,DEPState<DEPNode>>

	public DEPFFNNTrain(String[] args)
    {
        super(args);
    }


	@Override
	protected NLPConfig<DEPNode> createConfiguration(String filename)
	{
		return new DEPConfig(IOUtils.createFileInputStream(filename));
	}
	
	@Override
	protected Eval createEvaluator()
	{
		return new DEPEval(); //why use this eval over AccuracyEval.java
	}
	@Override
	protected NLPComponent<DEPNode,DEPState<DEPNode>> createComponent()
	{
		return new DEPParser<>(new StringModel(new MultinomialWeightVector()));
	}
	
	@Override
	protected FeatureTemplate<DEPNode,DEPState<DEPNode>> createFeatureTemplate()
	{
		switch (feature_template)
		{
		case 0: return new DEPFeatureTemplate0();
		case 1: return new DEPFeatureTemplate1();
		case 2: return new DEPFeatureTemplate2();
		default: throw new IllegalArgumentException("Unknown feature template: "+feature_template);
		}
	}
	
	static public void main(String[] args)
	{
		new DEPFFNNTrain(args).train();
	}

	@Override
	public void collect(TSVReader<DEPNode> reader, List<String> inputFiles,
			NLPComponent<DEPNode, DEPState<DEPNode>> component, NLPConfig<DEPNode> configuration) {
		//not exactly sure that the purpose of this method aws suppose to be
	}


}