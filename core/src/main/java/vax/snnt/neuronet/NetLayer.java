package vax.snnt.neuronet;

import java.util.LinkedList;
import java.util.List;

/**

 @author toor
 */
public class NetLayer implements Processable {
    private final List<Neuron> neurons = new LinkedList<>(); // TODO switch to indexes from Network array

    /*
     public NetLayer ( List<Neuron> neurons ) {
     this.neurons = neurons;
     }
     */

 /*
     public NetLayer ( NetLayer nextLayer ) {

     }
     */
    public NetLayer () {
    }

    public void addNeuron ( Neuron neuron ) {
        neurons.add( neuron );
    }

    @Override
    public void process () {
        for( Neuron neuron : neurons ) {
            neuron.process();
        }
    }

    public List<Neuron> getNeurons () {
        return neurons;
    }

}
