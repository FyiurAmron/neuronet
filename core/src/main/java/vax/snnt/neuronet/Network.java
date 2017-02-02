package vax.snnt.neuronet;

import java.util.ArrayList;
import java.util.List;
import java.util.*;
import java.util.Map.Entry;
import static java.lang.Double.isNaN;
import static java.lang.Math.abs;
import java.util.function.DoubleConsumer;

/**

 @author toor
 */
public class Network implements Processable {
    private final ArrayList<Neuron> neurons = new ArrayList<>();
    private transient final HashMap<Neuron, Integer> neuronMap = new HashMap<>(); // TODO proper deserialization for it
    private final HashMap<Integer, HashMap<Integer, Double>> neuronInputs = new HashMap<>();
    private final HashMap<Integer, HashMap<Integer, Double>> neuronOutputs = new HashMap<>();
    private final ArrayList<Layer> layers = new ArrayList<>(); // may be shared in general though

    private final HashSet<Integer> outputNeurons = new HashSet<>();

    private transient final HashMap<Integer, DoubleConsumer> outputNeuronConsumers = new HashMap<>();

    private transient final ArrayList<Double> outputs = new ArrayList<>();
    private transient double distL1, distL2;

    public Network () {
    }

    @SuppressWarnings( "unchecked" )
    public Network ( Network network ) {
        ArrayList<Neuron> networkNeurons = network.neurons;
        for( int i = 0, max = networkNeurons.size(); i < max; i++ ) {
            Neuron neuron = networkNeurons.get( i ).copy();
            neurons.add( neuron );
            neuronMap.put( neuron, i );
        }

        ArrayList<Layer> networkLayers = network.layers;
        for( int i = 0, max = networkLayers.size(); i < max; i++ ) {
            Layer layer = networkLayers.get( i ).copy();
            layers.add( layer );
        }
        //layers.addAll( network.layers ); // share layers, since they don't mutate

        outputNeurons.addAll( network.outputNeurons );

        // TODO replace with arrays maybe?
        for( Entry<Integer, HashMap<Integer, Double>> entry : network.neuronInputs.entrySet() ) {
            neuronInputs.put( entry.getKey(), (HashMap<Integer, Double>) entry.getValue().clone() );
        }
        for( Entry<Integer, HashMap<Integer, Double>> entry : network.neuronOutputs.entrySet() ) {
            neuronOutputs.put( entry.getKey(), (HashMap<Integer, Double>) entry.getValue().clone() );
        }
    }

    public Network copy () {
        return new Network( this );
    }

    public Network mutate ( /* Random random, */ double mutationFactorAmp, double mutationFactorShift, double mutationFactorWeight ) {
        for( Neuron neuron : neurons ) {
            TransferFunction itf = neuron.inputTransferFunction;
            itf.ampIn += Rng.rnd() * mutationFactorAmp;
            itf.ampOut += Rng.rnd() * mutationFactorAmp;
            itf.shiftIn += Rng.rnd() * mutationFactorShift;
            itf.shiftOut += Rng.rnd() * mutationFactorShift;

            TransferFunction otf = neuron.outputTransferFunction;
            otf.ampIn += Rng.rnd() * mutationFactorAmp;
            otf.ampOut += Rng.rnd() * mutationFactorAmp;
            otf.shiftIn += Rng.rnd() * mutationFactorShift;
            otf.shiftOut += Rng.rnd() * mutationFactorShift;

            for( Entry<Integer, HashMap<Integer, Double>> entry : neuronOutputs.entrySet() ) {
                int a = entry.getKey();
                for( Entry<Integer, Double> entry2 : entry.getValue().entrySet() ) {
                    int b = entry2.getKey();
                    double newValue = entry2.getValue() + Rng.rnd() * mutationFactorWeight;
                    entry2.setValue( newValue );
                    neuronInputs.get( b ).put( a, newValue );
                }
            }
        }
        return this; // for call chaining
    }

    public void addLayer ( vax.snnt.neuronet.Layer layer ) {
        layers.add( layer );
    }

    public int addNeuron ( Neuron neuron, vax.snnt.neuronet.Layer layer ) {
        int nr = neurons.size();
        neurons.add( neuron );
        neuronMap.put( neuron, nr );
        layer.addNeuron( nr );
        return nr;
    }

    public Neuron getNeuron ( int number ) {
        return neurons.get( number );
    }

    public int getNeuronNumber ( Neuron neuron ) {
        Integer integer = neuronMap.get( neuron );
        if ( integer == null ) {
            throw new IllegalArgumentException();
        }
        return integer;
        //return ( integer == null ) ? -1 : integer;
    }

    public void addOutputConsumer ( Neuron neuron, DoubleConsumer doubleConsumer ) {
        addOutputConsumer( getNeuronNumber( neuron ), doubleConsumer );
    }

    public void addOutputConsumer ( int neuronNr, DoubleConsumer doubleConsumer ) {
        outputNeuronConsumers.put( neuronNr, doubleConsumer );
    }

    /*
     public Layer<? extends Neuron> getInputLayer() {
     return layers.getFirst();
     }
     */
    public void addInputs ( double[] inputs ) {
        List<Integer> inputNeuronNrs = layers.get( 0 ).getNeurons();
        if ( inputNeuronNrs.size() != inputs.length ) {
            throw new IllegalArgumentException();
        }
        int i = 0;
        for( int j : inputNeuronNrs ) {
            neurons.get( j ).add( inputs[i] );
            i++;
        }
    }

    public void collectOutput ( double output ) {
        outputs.add( output );
    }

    public double[] getOutputs () {
        int count = outputs.size();
        double[] ret = new double[count];
        for( int i = 0; i < count; i++ ) {
            ret[i] = outputs.get( i );
        }
        return ret;
    }

    @Override
    public void process () {
        outputs.clear();
        for( vax.snnt.neuronet.Layer layer : layers ) {
            for( int i : layer.getNeurons() ) {
                Neuron neuron = neurons.get( i );
                neuron.preProcess();
                double pot = neuron.getPotential();
                HashMap<Integer, Double> neuronOutputMap = neuronOutputs.get( i );
                if ( neuronOutputMap != null ) {
                    for( Entry<Integer, Double> neuronOutput : neuronOutputMap.entrySet() ) {
                        neurons.get( neuronOutput.getKey() ).add( pot * neuronOutput.getValue() );
                    }
                }
                DoubleConsumer outputConsumer = outputNeuronConsumers.get( i );
                if ( outputConsumer != null ) {
                    outputConsumer.accept( pot );
                }
                if ( outputNeurons.contains( i ) ) {
                    collectOutput( pot );
                }
                neuron.postProcess();
            }
        }
    }

    public void addConnection ( Neuron inputNeuron, Neuron outputNeuron, double weight ) {
        addConnection( getNeuronNumber( inputNeuron ), getNeuronNumber( outputNeuron ), weight );
    }

    public void addConnection ( int inputNr, int outputNr, double weight ) {
        HashMap<Integer, Double> neuronInputMap = neuronInputs.get( outputNr );
        if ( neuronInputMap == null ) {
            neuronInputMap = new HashMap<>();
            neuronInputs.put( outputNr, neuronInputMap );
        }
        HashMap<Integer, Double> neuronOutputMap = neuronOutputs.get( inputNr );
        if ( neuronOutputMap == null ) {
            neuronOutputMap = new HashMap<>();
            neuronOutputs.put( inputNr, neuronOutputMap );
        }

        if ( neuronInputMap.containsKey( inputNr ) || neuronOutputMap.containsKey( outputNr ) ) {
            throw new UnsupportedOperationException( "connection already present & weight already set" );
        }

        neuronInputMap.put( inputNr, weight );
        neuronOutputMap.put( outputNr, weight );
    }

    /*
     public void addNeuronOutputs ( List<Neuron> ns, double[] weights ) {
     if ( ns.isEmpty() || ns.size() != weights.length ) {
     throw new IllegalArgumentException();
     }

     Iterator<Neuron> it = ns.listIterator();
     int i = 0;
     for( Neuron n = it.next(); it.hasNext(); n = it.next() ) {
     neuronOutputs.add( new NeuronOutput( n, weights[i] ) );
     i++;
     }
     }
     */
    public void calcOutputDist ( double[] expectedOutputs ) {
        int count = outputs.size();
        if ( expectedOutputs.length != count ) {
            throw new IllegalArgumentException();
        }

        for( int i = 0; i < count; i++ ) {
            double diff = outputs.get( i ) - expectedOutputs[i];
            distL1 += abs( diff );
            distL2 += diff * diff;
        }
        if ( isNaN( distL1 ) || isNaN( distL2 ) ) {
            throw new RuntimeException();
        }
    }

    public void resetOutputDist () {
        distL1 = 0;
        distL2 = 0;
    }

    public double getDistL1 () {
        return distL1;
    }

    public double getDistL2 () {
        return distL2;
    }

    public Iterable<Neuron> layerIterable ( vax.snnt.neuronet.Layer midLayer ) {
        return () -> {
            return new Iterator<Neuron>() {
                private int i = 0;

                @Override
                public boolean hasNext () {
                    return i < midLayer.getNeurons().size();
                }

                @Override
                public Neuron next () {
                    Neuron neuron = neurons.get( midLayer.getNeurons().get( i ) );
                    i++;
                    return neuron;
                }
            };
        };
    }
    // end

    public void addToOutputs ( Neuron neuron ) {
        addToOutputs( getNeuronNumber( neuron ) );
    }

    public void addToOutputs ( int neuronNr ) {
        outputNeurons.add( neuronNr );
    }
}
