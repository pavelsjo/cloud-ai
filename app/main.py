import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.preprocessing import image
import streamlit as st
import io
from PIL import Image

#funcions
def weighted_loss(y_true, y_pred):
        
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
            loss += K.mean((-1 * pos_weights[i] * y_true[:,i]*K.log(y_pred[:,i]+epsilon)) + (- 1 * neg_weights[i]*(1-y_true[:,i])*K.log(1-y_pred[:,i]+epsilon))) #complete this line
        return loss
# main app
st.title("Cloud AI - DEMO")
st.header("Architecure detector from cloud Oracle or Azure")

# image
uploaded = st.file_uploader("Please upload an image file", type=["png"])

if uploaded is not None:

    # load
    img = Image.open(uploaded)
    st.image(img, caption='Uploaded Image, await for predictions...', use_column_width=True)
    
    # pipe 1
    model_pipe_one = tf.keras.models.load_model('./models/pipe1-providers.h5')
    labels_provider = {0:'Azure', 1:'Oracle'}
    img = img.convert('RGB')
    img = img.resize((350,350), Image.NEAREST)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = model_pipe_one.predict(images, batch_size=10)
    st.subheader(f"CLOUD PROVIDER DETECTED: {labels_provider[prediction[0][0]]}")

    if prediction == 1:
        labels = ['/gfs-data', 'API gateway', 'ATG API tier', 'Access and interpretation', 'Administration tier', 'Application server', 'Attach each FortiGate virtual network interface card (VNIC) into each spoke VCN.', 'Audit logs', 'Autonomous Data Warehouse', 'Autonomous Database System', 'Autonomous Transaction Processing', 'Autonomous data warehouse', 'Autonomous database', 'Autonomous database, with a private endpoint', 'Autoscaling', 'Availability domain', 'Availability domains', 'Bastion host', 'Block storage', 'Block volume', 'Block volumes', 'Border Gateway Protocol (BGP) routing', 'Bot-access VM', 'Bots', 'CES node', 'Client devices', 'Client node', 'Client nodes', 'Compartment', 'Compute', 'Compute and application servers', 'Compute instance', 'Compute instances', 'Container Engine for Kubernetes', 'Custom logs', 'Customer Service Center (CSC) tier', 'Customer-premises equipment (CPE)', 'DNS', 'Data nodes', 'Data persistence platform (curated information layer)', 'Data refinery', 'Database', 'Database System', 'Database servers', 'Database shards', 'Database system', 'Databases', 'Dynamic Routing Gateway', 'Dynamic routing gateway', 'Dynamic routing gateway (DRG)', 'Endeca MDEX tier', 'Event monitoring and alerting', 'Events', 'Events and Functions', 'ExpressRoute', 'FastConnect', 'Fault domain', 'Fault domains', 'File Storage', 'File storage', 'Firewall', 'Functions', 'GFS-nodes', 'HPC cluster node', 'Head node', 'IPv4 and IPv6', 'Inference server', 'Instance configuration', 'Instance pool', 'Integration instance', 'Internet', 'Internet Gateway', 'Internet Protocol Security (IPSec)', 'Internet gateway', 'Java Microservices', 'Jenkins', 'Jenkins agent instances', 'Jenkins master instance', 'Key management', 'Kibana', 'Leverage transit-routing with local peering gateways (LPG) to connect spoke VCNs with the hub VCN.', 'Load Balancer', 'Load balancer', 'Local peering gateway (LPG)', 'Logging', 'Logging addon for Splunk', 'Lustre clients', 'MGMT GUI node', 'Management Server (MGS)', 'Management server', 'Master nodes', 'Metadata Server (MDS)', 'Metadata service', 'Microsoft Azure Components', 'MongoDB node', 'Monitoring', 'MySQL Database', 'NAT gateway', 'NSD server', 'Network', 'Network security group', 'Network security groups', 'Network security groups (NSG)', 'Notifications', 'Object Storage', 'Object Storage Servers (OSS)', 'Object Storage service', 'Object storage', 'Observability and Tracing', 'Observers', 'On-premises deployment', 'On-premises network', 'Ops Manager', 'Oracle Analytics Cloud', 'Oracle Application Express (APEX)', 'Oracle Cloud Infrastructure Components', 'Oracle Cloud Infrastructure Data Catalog', 'Oracle Cloud Infrastructure Data Flow', 'Oracle Functions', 'Oracle Storage Gateway', 'Oracle WebLogic Server cluster', 'Oracle WebLogic Server domain', 'Oracle services network', 'Orchestrator', 'Policy', 'Polyglot Microservices', 'Primary and standby DB systems', 'Primary and standby SBC nodes', 'Primary and standby shard catalogs', 'Private DNS Resolver', 'Private peering', 'Public peering', 'Region', 'Regions', 'Registry', 'Relational Junction instance', 'Route table', 'Route tables', 'SGD gateways', 'SGD servers', 'SQL Server', 'Schedule-based autoscaling', 'Security List', 'Security list', 'Security lists', 'Service Mesh with Oracle Container Engine for Kubernetes', 'Service connectors', 'Service gateway', 'Service logs', 'Shard directors', 'Static routing', 'Storefront tier', 'Streaming', 'Subnets', 'Tenancy', 'Third-party backup application', 'Tomcat servers', 'Traditional Applications', 'Training node', 'Tunnel', 'UiPath Studio on User Machine', 'User Application VM - Oracle Cloud Infrastructure Compute', 'User application VM', 'User demo-finance-user, in group demo-finance-users', 'User demo-hr-user, in group demo-hr-users', 'User demo-it-user, in group demo-it-users', 'User demo-marketing-user, in group demo-marketing-users', 'VCN', 'VPN Connect', 'Varnish Enterprise (VE1 and VE2) nodes', 'Virtual circuit', 'Virtual cloud network (VCN)', 'Virtual cloud network (VCN) and subnet', 'Virtual cloud network (VCN) and subnets', 'Virtual cloud network and subnets', 'Virtual cloud networks (VCN) and subnets', 'Virtual machines', 'Virtual machines (VMs)', 'Virtual network (VNet)', 'Virtual network gateway', 'Virtual network interface card (VNIC)', 'Visualization node', 'Web server', 'WebLogic Kubernetes operator', 'WebLogic cluster', 'WebLogic domain', 'WildFly servers', 'Work Place VM']
        model_pipe_two_oracle = tf.keras.models.load_model('./models/pipe2-oracle.h5', custom_objects={'weighted_loss': weighted_loss})
        classes = model_pipe_two_oracle.predict(images)
        # json output
        full_prediction = dict(zip(labels,classes[0]))
        filter_prediction = {k: v for k, v in full_prediction.items() if v != 0 }
        tags = [k for k,v in filter_prediction.items()]
        st.write(f"TAGS: {tags}")
